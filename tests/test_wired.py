from pathlib import Path

import mlx.core as mx

from laguna_mlx import wire_resident


def test_wired_limit_only_for_resident_models():
    before = mx.set_wired_limit(0)
    try:
        wire_resident(False)  # streaming: leave the rest of the machine alone
        assert mx.set_wired_limit(0) == 0
        wire_resident(True)
        assert mx.set_wired_limit(0) == mx.device_info()["max_recommended_working_set_size"]
    finally:
        mx.set_wired_limit(before)


def test_scripts_go_through_the_helper():
    raw = [p.name for p in Path("scripts").glob("*.py") if "set_wired_limit" in p.read_text()]
    assert raw == []


def test_mx_without_wired_limit_ignores_only_that_call():
    from laguna_mlx import mx_without_wired_limit

    before = mx.set_wired_limit(0)
    try:
        m = mx_without_wired_limit()
        m.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        assert mx.set_wired_limit(0) == 0 and m.device_info is mx.device_info
    finally:
        mx.set_wired_limit(before)


def test_serve_streaming_keeps_mlx_server_off_the_wired_limit(monkeypatch):
    import mlx_lm.server as server
    from scripts import serve_streaming

    seen = {}
    monkeypatch.setattr(server, "main", lambda: seen.update(mx=server.mx))
    monkeypatch.setattr(server, "mx", server.mx)  # restored after the test
    monkeypatch.setattr("sys.argv", ["serve"])
    serve_streaming.main()
    before = mx.set_wired_limit(0)
    try:
        seen["mx"].set_wired_limit(1 << 30)
        assert mx.set_wired_limit(0) == 0
    finally:
        mx.set_wired_limit(before)
