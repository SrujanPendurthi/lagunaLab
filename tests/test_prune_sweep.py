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
