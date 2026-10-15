import json

import numpy as np
import pytest
from transformers import AutoTokenizer

from scripts import bench_kv
from tests.tiny import mlx_only_model

MESSAGES = [
    {"role": "user", "content": "list files"},
    {"role": "assistant", "content": "There are two files, a.py and b.py."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("models/Laguna-XS-2.1")


@pytest.fixture
def traces(tmp_path):
    out = []
    for name, extra in (("short", ""), ("long", " Please also describe each of them in detail.")):
        f = tmp_path / f"{name}.jsonl"
        msgs = [{**MESSAGES[0], "content": MESSAGES[0]["content"] + extra}, MESSAGES[1]]
        f.write_text(json.dumps({"path": "/v1/chat/completions", "request": {"messages": msgs},
                                 "status": 200, "response": ""}))
        out.append(f)
    return out


def test_sweep_rows_summary_and_resume(tok, traces, tmp_path, monkeypatch):
    m = mlx_only_model(head_dim=64)
    refs_dir, out = tmp_path / "refs", tmp_path / "kv.json"
    refs = bench_kv.build_refs(m, tok, traces, refs_dir, cap=4096)
    bench_kv.sweep(m, "resident", (None, 8, 4), refs, out, decode_n=4)
    res = json.loads(out.read_text())
    assert set(res["rows"]) == {f"resident/{b}/{s}" for b in ("plain", "8", "4") for s in ("short", "long")}
    # decode numbers only on the longest session
    assert "decode_tok_s" in res["rows"]["resident/8/long"] and "decode_tok_s" not in res["rows"]["resident/8/short"]
    s = res["summary"]
    assert abs(s["resident/plain"]["kl_mean"]) < 1e-5 and s["resident/plain"]["top1"] == 1.0
    assert s["resident/4"]["kv_gb"] < s["resident/8"]["kv_gb"] < s["resident/plain"]["kv_gb"]
    assert s["resident/plain"]["fits"] and s["resident/plain"]["passes"]

    # resume: nothing is recomputed, the file is unchanged
    def boom(*a, **k):
        raise AssertionError("recomputed a finished row")

    monkeypatch.setattr(bench_kv, "score", boom)
    before = out.read_text()
    bench_kv.sweep(m, "resident", (None, 8, 4), refs, out, decode_n=4)
    assert out.read_text() == before


def test_truncated_reference_is_refused(tok, traces, tmp_path):
    m = mlx_only_model(head_dim=64)
    full = bench_kv.build_refs(m, tok, traces, tmp_path / "full", cap=4096)
    n = len(np.load(full[-1])["ids"])  # longest session
    with pytest.raises(ValueError, match="truncated"):
        bench_kv.build_refs(m, tok, traces, tmp_path / "refs", cap=n - 2)


def test_sweep_stops_before_longer_sessions_when_peak_too_high(tok, traces, tmp_path, monkeypatch):
    # sessions run shortest first, so a memory blow-up on a short one must stop the run
    # before the longer ones (an 8-bit row at 78k context once hung the machine)
    m = mlx_only_model(head_dim=64)
    refs = bench_kv.build_refs(m, tok, traces, tmp_path / "refs", cap=4096)
    out = tmp_path / "kv.json"
    monkeypatch.setattr(bench_kv, "PEAK_ABORT_GB", 1e-6)
    with pytest.raises(SystemExit, match="peak"):
        bench_kv.sweep(m, "resident", (8,), refs, out, decode_n=4)
    assert list(json.loads(out.read_text())["rows"]) == ["resident/8/short"]  # recorded, then stopped
    with pytest.raises(SystemExit, match="peak"):  # a resume must not skip past the recorded blow-up
        bench_kv.sweep(m, "resident", (8,), refs, out, decode_n=4)
    assert list(json.loads(out.read_text())["rows"]) == ["resident/8/short"]


def test_streaming_label_keeps_old_rows_and_separates_new_configs():
    # the original run (32 slots, uncapped) keeps its key; other configs get their own rows
    assert bench_kv.streaming_label(32, None) == "streaming"
    assert bench_kv.streaming_label(64, 1.0) == "streaming-64slots-cap1.0gb"
    assert bench_kv.streaming_label(32, 1.0) == "streaming-32slots-cap1.0gb"
