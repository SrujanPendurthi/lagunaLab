"""Tests for scripts/overnight.sh -- CPU-only, no real models.

PATH is shimmed with fake `python3` (logs its argv, exits 0/1 per
FAKE_PY_FAIL), `sysctl` (prints FAKE_SWAP_OUTPUT), and `pgrep` (matches per
FAKE_PGREP_MATCH), so nothing real ever runs. Each test copies
scripts/overnight.sh into a throwaway tmp_path/scripts/ dir so the script's
own `cd "$(dirname "$0")/.."` lands in tmp_path -- results/overnight/ never
touches the real project tree.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERNIGHT_SH = REPO_ROOT / "scripts" / "overnight.sh"

NORMAL_SWAP = "vm.swapusage: total = 2048.00M  used = 606.50M  free = 1441.62M  (encrypted)"
HIGH_SWAP = "vm.swapusage: total = 32768.00M  used = 15000.00M  free = 17768.00M  (encrypted)"
HIGH_SWAP_G = "vm.swapusage: total = 32.00G  used = 11.50G  free = 20.50G  (encrypted)"
LOW_SWAP_G = "vm.swapusage: total = 32.00G  used = 1.20G  free = 30.80G  (encrypted)"
GARBAGE_SWAP = "vm.swapusage: kaboom"

FULL_QUEUE = [
    "-m scripts.diag_dflash --target both --skip-existing",
    "-m scripts.prune_sweep calibrate",
    "-m scripts.prune_sweep refs",
    "-m scripts.prune_sweep sweep",
    "-m scripts.bench_lookup --skip-existing",
]


def _write_exe(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def make_repo(tmp_path):
    """tmp_path/scripts/overnight.sh, so the script's own cd lands here."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy(OVERNIGHT_SH, scripts_dir / "overnight.sh")
    return tmp_path


def make_bins(tmp_path, py_log):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    _write_exe(bindir / "python3", f"""#!/usr/bin/env bash
echo "$@" >> "{py_log}"
if [ -n "${{FAKE_PY_FAIL:-}}" ] && [[ "$*" == *"$FAKE_PY_FAIL"* ]]; then
    exit 1
fi
exit 0
""")
    _write_exe(bindir / "sysctl", """#!/usr/bin/env bash
if [ "$1" = "vm.swapusage" ]; then
    echo "$FAKE_SWAP_OUTPUT"
    exit 0
fi
exit 1
""")
    _write_exe(bindir / "pgrep", """#!/usr/bin/env bash
if [ "${FAKE_PGREP_MATCH:-0}" = "1" ]; then
    echo 12345
    exit 0
fi
exit 1
""")
    return bindir


def run_overnight(repo, bindir, env_extra=None):
    env = os.environ.copy()
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env.setdefault("FAKE_SWAP_OUTPUT", NORMAL_SWAP)
    env.setdefault("FAKE_PGREP_MATCH", "0")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(repo / "scripts" / "overnight.sh")],
        cwd=repo, env=env, capture_output=True, text=True, timeout=30,
    )


def invocations(py_log):
    if not py_log.exists():
        return []
    return [line.strip() for line in py_log.read_text().splitlines() if line.strip()]


def queue_log_text(repo):
    p = repo / "results" / "overnight" / "queue.log"
    return p.read_text() if p.exists() else ""


# --------------------------------------------------------------- happy path

def test_full_queue_runs_in_order(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir)

    assert result.returncode == 0
    assert invocations(py_log) == FULL_QUEUE
    log = queue_log_text(repo)
    for name in ("s1", "p2_calib", "p2_refs", "p2_sweep", "s4"):
        assert f"START {name}" in log
        assert f"END {name} exit=0" in log


def test_per_step_logs_written(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    run_overnight(repo, bindir)

    overnight_dir = repo / "results" / "overnight"
    for name in ("s1", "p2_calib", "p2_refs", "p2_sweep", "s4"):
        assert (overnight_dir / f"{name}.log").exists()


# --------------------------------------------------------- chain failure

def test_calibrate_failure_stops_chain_but_not_s1_s4(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir, {"FAKE_PY_FAIL": "prune_sweep calibrate"})

    assert result.returncode == 0
    assert invocations(py_log) == [
        "-m scripts.diag_dflash --target both --skip-existing",
        "-m scripts.prune_sweep calibrate",
        "-m scripts.bench_lookup --skip-existing",
    ]
    log = queue_log_text(repo)
    assert "END p2_calib exit=1" in log
    assert "SKIP p2_refs" in log
    assert "SKIP p2_sweep" in log
    assert "END s1 exit=0" in log
    assert "END s4 exit=0" in log


def test_calibrate_skipped_when_stats_exist_but_chain_continues(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "results").mkdir()
    (repo / "results" / "expert_stats.npz").write_bytes(b"")
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir)

    assert result.returncode == 0
    assert invocations(py_log) == [
        "-m scripts.diag_dflash --target both --skip-existing",
        "-m scripts.prune_sweep refs",
        "-m scripts.prune_sweep sweep",
        "-m scripts.bench_lookup --skip-existing",
    ]
    log = queue_log_text(repo)
    assert "SKIP p2_calib" in log
    assert "already exists" in log


def test_refs_failure_stops_sweep_only(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir, {"FAKE_PY_FAIL": "prune_sweep refs"})

    assert result.returncode == 0
    assert invocations(py_log) == [
        "-m scripts.diag_dflash --target both --skip-existing",
        "-m scripts.prune_sweep calibrate",
        "-m scripts.prune_sweep refs",
        "-m scripts.bench_lookup --skip-existing",
    ]
    log = queue_log_text(repo)
    assert "END p2_refs exit=1" in log
    assert "SKIP p2_sweep" in log


def test_s1_failure_does_not_block_chain_or_s4(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir, {"FAKE_PY_FAIL": "diag_dflash"})

    assert result.returncode == 0
    assert invocations(py_log) == FULL_QUEUE
    log = queue_log_text(repo)
    assert "END s1 exit=1" in log
    assert "END p2_calib exit=0" in log
    assert "END p2_refs exit=0" in log
    assert "END p2_sweep exit=0" in log
    assert "END s4 exit=0" in log


# --------------------------------------------------------------- STOP file

def test_stop_file_skips_remaining_steps(tmp_path):
    repo = make_repo(tmp_path)
    overnight_dir = repo / "results" / "overnight"
    overnight_dir.mkdir(parents=True)
    (overnight_dir / "STOP").touch()
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir)

    assert result.returncode == 0
    assert invocations(py_log) == []
    log = queue_log_text(repo)
    assert "SKIP s1: STOP file present" in log
    assert "SKIP p2_calib: STOP file present" in log
    # p2_refs/p2_sweep never ran calibrate/refs -- the skip reason must be
    # the halt, not the generic "predecessor did not succeed" text (that
    # would wrongly imply calibrate/refs ran and failed).
    assert "SKIP p2_refs: STOP file present" in log
    assert "SKIP p2_sweep: STOP file present" in log
    assert "did not succeed" not in log
    assert "SKIP s4: STOP file present" in log


# --------------------------------------------------------------- swap

def test_high_swap_skips_remaining_steps(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir, {"FAKE_SWAP_OUTPUT": HIGH_SWAP})

    assert result.returncode == 0
    assert invocations(py_log) == []
    log = queue_log_text(repo)
    assert "SKIP s1" in log
    assert "swap" in log.lower()
    # Same halt-reason-wins requirement as the STOP-file case.
    assert "SKIP p2_refs" in log and "swap" in log
    assert "SKIP p2_sweep" in log and "swap" in log
    assert "did not succeed" not in log
    assert "SKIP s4" in log


def test_high_swap_g_unit_skips_remaining_steps(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir, {"FAKE_SWAP_OUTPUT": HIGH_SWAP_G})

    assert result.returncode == 0
    assert invocations(py_log) == []
    log = queue_log_text(repo)
    assert "SKIP s1" in log
    assert "swap" in log.lower()


def test_low_swap_g_unit_runs_normally(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir, {"FAKE_SWAP_OUTPUT": LOW_SWAP_G})

    assert result.returncode == 0
    assert invocations(py_log) == FULL_QUEUE


def test_unparseable_swap_fails_closed(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir, {"FAKE_SWAP_OUTPUT": GARBAGE_SWAP})

    assert result.returncode == 0
    assert invocations(py_log) == []
    log = queue_log_text(repo)
    assert "SKIP s1" in log
    assert "parse" in log.lower()


# --------------------------------------------------------------- heavy process

def test_heavy_process_aborts_before_any_step(tmp_path):
    repo = make_repo(tmp_path)
    py_log = tmp_path / "py.log"
    bindir = make_bins(tmp_path, py_log)

    result = run_overnight(repo, bindir, {"FAKE_PGREP_MATCH": "1"})

    assert result.returncode != 0
    assert invocations(py_log) == []
    log = queue_log_text(repo)
    assert "ABORT" in log
