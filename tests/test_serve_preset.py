import mlx_lm.server as server
import pytest

from scripts import serve_streaming as ss


@pytest.fixture
def run(monkeypatch):
    """main() with the server stubbed out; returns (argv the server saw, module config)."""
    for name, v in (("pack", ss.PACK), ("slots", 32), ("cache_limit_gb", 1.0)):
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
