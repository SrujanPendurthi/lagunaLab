"""Run pool on SWE-bench Lite tasks; the proxy captures every model request.

Wiring (deviates from the original brief): ollama serves on its default port
11434. The proxy listens on 11435 and forwards to ollama at 11434. pool is
pointed at the proxy via env vars, per the pool README's "OpenAI-compatible
API" / standalone-mode section:
  POOLSIDE_STANDALONE_BASE_URL=http://127.0.0.1:11435   (bare URL; pool appends /v1 itself)
  POOLSIDE_API_KEY=EMPTY
  POOLSIDE_STANDALONE_MODEL=<ollama tag, e.g. laguna-xs-2.1>
  POOLSIDE_STANDALONE_CONTEXT_LENGTH=32768

Prereqs (separate terminal): OLLAMA_CONTEXT_LENGTH=32768 ollama serve
python3 -m scripts.gen_traces --n 50      # resumable: skips finished tasks
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from replay.session import load_session

PROMPT = ("Resolve the following GitHub issue in the repository in the current directory. "
          "Make the minimal code change needed.\n\n{issue}")

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50)
p.add_argument("--work", default=str(Path.home() / "laguna-work"))
p.add_argument("--traces", default="traces")
p.add_argument("--timeout", type=int, default=3600)
p.add_argument("--proxy-port", type=int, default=11435)
p.add_argument("--ollama-port", type=int, default=11434)
p.add_argument("--model", default="laguna-xs-2.1")
p.add_argument("--context-length", type=int, default=32768)
a = p.parse_args()

work, traces = Path(a.work), Path(a.traces)
traces.mkdir(exist_ok=True)
df = pd.read_parquet(glob.glob("data/swe-bench-lite/data/test-*.parquet")[0])
df = df.sort_values("instance_id").head(a.n)

pool_env = {
    **os.environ,
    "POOLSIDE_STANDALONE_BASE_URL": f"http://127.0.0.1:{a.proxy_port}",
    "POOLSIDE_API_KEY": "EMPTY",
    "POOLSIDE_STANDALONE_MODEL": a.model,
    "POOLSIDE_STANDALONE_CONTEXT_LENGTH": str(a.context_length),
}

for row in df.itertuples():
    out = traces / f"{row.instance_id}.jsonl"
    if out.exists():
        continue
    mirror = work / "_repos" / row.repo.replace("/", "__")
    if not mirror.exists():
        subprocess.run(["git", "clone", "--quiet", f"https://github.com/{row.repo}.git", str(mirror)], check=True)
    wt = work / row.instance_id
    if not wt.exists():
        subprocess.run(["git", "-C", str(mirror), "worktree", "add", "--detach", str(wt), row.base_commit], check=True)
    partial = out.with_suffix(".partial")
    proxy = subprocess.Popen([sys.executable, "-m", "scripts.proxy", "--listen", str(a.proxy_port),
                              "--upstream", f"http://127.0.0.1:{a.ollama_port}", "--out", str(partial)])
    time.sleep(1)
    t0 = time.time()
    try:
        r = subprocess.run(["pool", "exec", "-p", PROMPT.format(issue=row.problem_statement),
                            "-o", "json", "--unsafe-auto-allow"],
                           cwd=wt, timeout=a.timeout, capture_output=True, text=True, env=pool_env)
        (traces / f"{row.instance_id}.pool.json").write_text(r.stdout)
        status = f"exit {r.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
    finally:
        proxy.terminate()
        proxy.wait()
    if partial.exists():
        partial.rename(out)
    print(f"{row.instance_id}: {status} in {time.time() - t0:.0f}s", flush=True)

summary = []
for f in sorted(traces.glob("*.jsonl")):
    try:
        msgs, _ = load_session(f)
        summary.append({"session": f.stem, "messages": len(msgs),
                        "assistant_turns": sum(m["role"] == "assistant" for m in msgs)})
    except ValueError as e:
        summary.append({"session": f.stem, "error": str(e)})
Path("results/traces_summary.json").write_text(json.dumps(summary, indent=1))
