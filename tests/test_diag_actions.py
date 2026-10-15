import json

from scripts import diag_actions


def test_flags_route_refs_models_and_output(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(diag_actions, "wire_resident", lambda r: None)
    monkeypatch.setattr(diag_actions, "run",
                        lambda path, refs: seen.append((path, refs)) or {"exact": 1.0, "same_tool": 1.0})
    out = tmp_path / "diag.json"
    diag_actions.main(["--pruned", "P", "--base", "B", "--refs", "R", "--out", str(out)])
    assert seen == [("B", "R"), ("P", "R")] and set(json.loads(out.read_text())) == {"base", "pruned"}


def test_defaults_unchanged():
    assert diag_actions.BASE == "models/Laguna-XS-2.1-mlx-4bit"
    assert str(diag_actions.AREFS) == "refs/prune_actions"
