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
