"""End-to-end dry run of scripts.prune_sweep on the tiny model: plumbing only, no real model."""
import json

import pytest
from transformers import AutoTokenizer

import scripts.prune_sweep as ps
from tests.tiny import TINY, mlx_model, torch_model

E, K = TINY["num_experts"], TINY["num_experts_per_tok"]
MOE_LAYERS = [i for i, t in enumerate(TINY["layer_types"]) if i not in TINY["mlp_only_layers"]]


def _session(path, user, reply):
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    path.write_text(json.dumps({"path": "/v1/chat/completions", "request": {"messages": messages},
                                 "status": 200, "response": ""}))


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("models/Laguna-XS-2.1")


@pytest.fixture
def model():
    return mlx_model(torch_model())


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point every module path at tmp_path and shrink the grid/min_keep for 8 experts."""
    traces = tmp_path / "traces"
    traces.mkdir()
    calib = []
    for i in range(3):
        f = traces / f"astropy__{i}.jsonl"
        _session(f, f"calib session {i}: list the files", f"Here are the results for case {i}.")
        calib.append(str(f))
    heldout = []
    for i in range(2):
        f = traces / f"django__{i}.jsonl"
        _session(f, f"heldout session {i}: run the tests", f"Tests pass for held-out case {i}.")
        heldout.append(str(f))

    monkeypatch.setattr(ps, "CALIB", calib)
    monkeypatch.setattr(ps, "HELDOUT", heldout)
    monkeypatch.setattr(ps, "CALIB_EVAL", calib[:1])
    monkeypatch.setattr(ps, "REFS", tmp_path / "refs" / "prune")
    monkeypatch.setattr(ps, "OUT", tmp_path / "pruning.json")
    monkeypatch.setattr(ps, "STATS_PATH", tmp_path / "expert_stats.npz")
    monkeypatch.setattr(ps, "STATS_META_PATH", tmp_path / "expert_stats_meta.json")
    monkeypatch.setattr(ps, "GLOBAL_MIN_KEEP", 2)  # tiny model has 8 experts/layer, not 128+
    monkeypatch.setattr(ps, "GRID", [("reap", "uniform", 0.25), ("random", "uniform", 0.5),
                                      ("reap", "global", 0.5)])
    return tmp_path


def test_calibrate_refs_sweep_end_to_end(model, tok, wired):
    cap = 512

    ps.calibrate(model, tok, cap)
    assert ps.STATS_PATH.exists() and ps.STATS_META_PATH.exists()
    meta = json.loads(ps.STATS_META_PATH.read_text())
    assert meta["sessions"] == [f"astropy__{i}" for i in range(3)]

    ps.refs(model, tok, cap)
    ref_files = sorted(p.name for p in ps.REFS.glob("*.npz"))
    assert ref_files == sorted({f"astropy__{i}.npz" for i in range(1)}
                                | {f"django__{i}.npz" for i in range(2)})

    ps.sweep(model)
    assert ps.OUT.exists()
    assert not ps.OUT.with_suffix(".tmp").exists()  # tmp+replace leaves no stray file behind
    data = json.loads(ps.OUT.read_text())
    rows = {(r["criterion"], r["allocation"], r["ratio"]): r for r in data["rows"]}
    assert len(rows) == 4  # baseline + 3 grid configs

    baseline = rows[("none", "uniform", 0.0)]
    assert baseline["kept_per_layer"] is None
    assert baseline["heldout"]["kl_mean"] < 1e-5
    assert baseline["heldout"]["top1"] == 1.0

    uniform = rows[("reap", "uniform", 0.25)]
    assert uniform["kept_per_layer"] == {str(i): 6 for i in MOE_LAYERS}  # 8 - round(0.25*8)

    uniform2 = rows[("random", "uniform", 0.5)]
    assert uniform2["kept_per_layer"] == {str(i): 4 for i in MOE_LAYERS}  # 8 - round(0.5*8)

    glob_row = rows[("reap", "global", 0.5)]
    total = len(MOE_LAYERS) * E
    budget = total - round(0.5 * total)
    assert sum(glob_row["kept_per_layer"].values()) == budget
    assert all(v >= 2 for v in glob_row["kept_per_layer"].values())  # respects GLOBAL_MIN_KEEP

    # resumability: rerunning sweep must not redo or duplicate finished rows
    before = json.loads(ps.OUT.read_text())
    ps.sweep(model)
    after = json.loads(ps.OUT.read_text())
    assert before == after


def test_grid_priority_order():
    """The real (unpatched) GRID: baseline first, then reap/random/frequency uniform (all
    ratios), then reap global, then gate uniform last -- per the controller's priority ruling."""
    ratios = ps.RATIOS
    expected = ([("reap", "uniform", r) for r in ratios]
                + [("random", "uniform", r) for r in ratios]
                + [("frequency", "uniform", r) for r in ratios]
                + [("reap", "global", r) for r in (0.25, 0.5)]
                + [("gate", "uniform", r) for r in ratios])
    assert ps.GRID == expected


def test_baseline_sanity_check_saves_row_then_exits(model, tok, wired, monkeypatch):
    cap = 512
    ps.calibrate(model, tok, cap)
    ps.refs(model, tok, cap)

    def fake_score(model, ref_path, chunk=2048):
        return {"tokens": 10, "assistant_tokens": 10, "truncated": False,
                "kl_top64_mean": 1.0, "kl_top64_p99": 1.0, "top1_agree": 0.0,
                "peak_mem_gb": 0.0, "tok_per_s": 1.0}

    monkeypatch.setattr(ps, "score", fake_score)

    with pytest.raises(SystemExit, match="baseline sanity check failed"):
        ps.sweep(model)

    assert ps.OUT.exists()  # the bad baseline row was still saved
    data = json.loads(ps.OUT.read_text())
    assert len(data["rows"]) == 1
    assert data["rows"][0]["criterion"] == "none"
    assert data["rows"][0]["heldout"]["kl_mean"] == 1.0


def test_stale_stamp_refuses_resume(model, tok, wired):
    cap = 512
    ps.calibrate(model, tok, cap)
    ps.refs(model, tok, cap)
    ps.sweep(model)
    good = json.loads(ps.OUT.read_text())

    # recalibrate on fewer sessions: same paths, genuinely different stats.npz content/hash
    old_calib = list(ps.CALIB)
    ps.CALIB = old_calib[:1]
    ps.calibrate(model, tok, cap)

    with pytest.raises(SystemExit, match="move it aside"):
        ps.sweep(model)

    # pruning.json from the earlier, valid run must be untouched
    assert json.loads(ps.OUT.read_text()) == good
    ps.CALIB = old_calib
