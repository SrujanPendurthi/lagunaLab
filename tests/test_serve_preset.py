import mlx_lm.server as server
import pytest

from scripts import serve_streaming as ss


@pytest.fixture
def run(monkeypatch):
    """main() with the server stubbed out; returns (argv the server saw, module config)."""
    for name, v in (("pack", ss.PACK), ("slots", 32), ("cache_limit_gb", 1.0), ("fused_decode", False)):
        monkeypatch.setattr(ss, name, v)
    monkeypatch.setattr(server, "mx", server.mx)
    monkeypatch.setattr(server, "load", server.load)
    monkeypatch.setattr(server, "main", lambda: None)

    def go(*args):
        argv = ["serve", *args]
        monkeypatch.setattr("sys.argv", argv)
        ss.main()
        return argv, (ss.pack, ss.slots, ss.cache_limit_gb)
    return go


def test_laptop16_resolves_to_the_part3_answer_and_prints_it(run, capsys):
    argv, cfg = run("--preset", "laptop16", "--port", "8081")
    assert cfg == ("models/Laguna-XS-2.1-pruned-50-packed", 64, 1.0)
    assert argv[argv.index("--model") + 1] == "models/Laguna-XS-2.1-pruned-50-packed"
    assert argv[argv.index("--max-tokens") + 1] == "4096" and "--preset" not in argv
    out = capsys.readouterr().out
    assert "pruned-50-packed" in out and "slots=64" in out and "cache_limit_gb=1.0" in out and "kv=plain" in out


def test_explicit_flags_beat_the_preset(run):
    argv, cfg = run("--preset", "laptop16", "--slots", "32", "--max-tokens", "8192")
    assert cfg[1] == 32 and argv[argv.index("--max-tokens") + 1] == "8192"
    assert argv.count("--max-tokens") == 1


def test_defaults_unchanged_without_preset(run):
    argv, cfg = run()
    assert cfg == ("models/Laguna-XS-2.1-packed", 32, 1.0) and "--max-tokens" not in argv


def test_fused_decode_flag_switches_the_cache(run, monkeypatch, capsys):
    from laguna_mlx.cache import DequantKVCache
    from tests.tiny import mlx_only_model

    argv, _ = run("--preset", "laptop16", "--fused-decode")
    assert ss.fused_decode and "--fused-decode" not in argv and "kv=q8-fused" in capsys.readouterr().out
    m = mlx_only_model(head_dim=128)
    monkeypatch.setattr(ss, "load_streaming", lambda *a, **k: (m, type("T", (), {"eos_token_ids": [2, 24]})()))
    monkeypatch.setattr(ss, "pack", "models/Laguna-XS-2.1-pruned-50-packed")
    model, _ = ss._load()
    assert any(isinstance(c, DequantKVCache) and c.fused_decode for c in model.make_cache())


def test_fused_decode_off_by_default(run, capsys):
    run("--preset", "laptop16")
    assert not ss.fused_decode and "kv=plain" in capsys.readouterr().out
