"""Serve the SSD-streaming Laguna model through mlx-lm's OpenAI-compatible server.

python3 -m scripts.serve_streaming --slots 32 --port 8081 \
    --prompt-cache-size 6 --prompt-cache-bytes 4294967296 --max-tokens 4096
mlx-lm infers the GLM-4.7 tool parser from Laguna's chat template (<arg_key> tags).

--prompt-cache-size 2 (the original plan) evicts the growing conversation's
own KV cache entry under the assistant/user/system LRU split before the next
turn can reuse it, forcing a full re-prefill every turn; 6 (bounded to 4 GiB
via --prompt-cache-bytes) gives enough headroom for it to survive. --max-tokens
must be raised from mlx_lm.server's default (512): pool never sends its own
max_tokens, and a completion truncated at 512 makes pool abort the whole run.
See Task 8b report for the evidence (cached_tokens tracking prior total_tokens
turn over turn once these were set).
"""
import json
import sys
from pathlib import Path

import mlx_lm.server as server

import laguna_mlx  # noqa: F401
from laguna_mlx.streaming import load_streaming

PACK = "models/Laguna-XS-2.1-packed"
slots = 32


class _DropKeepalive:
    """Swallow mlx_lm.server's SSE keepalive comment lines (": keepalive N/M\\n\\n"),
    sent during long prefill to keep the connection alive. pool's OpenAI streaming
    client errors with "unexpected end of JSON input" the instant one arrives (it
    doesn't skip SSE comment lines per spec) and then retries the whole request,
    re-prefilling from scratch every time. Real data lines are untouched.
    """

    def __init__(self, raw):
        self.raw = raw

    def write(self, data):
        if data.lstrip(b"\r\n").startswith(b":"):
            return len(data)
        return self.raw.write(data)

    def flush(self):
        return self.raw.flush()

    def close(self):
        return self.raw.close()

    @property
    def closed(self):
        return self.raw.closed


_orig_setup = server.APIHandler.setup


def _setup(self):
    _orig_setup(self)
    self.wfile = _DropKeepalive(self.wfile)


server.APIHandler.setup = _setup


def _load(*a, **k):
    model, tok = load_streaming(PACK, slots)
    # generation_config.json's eos_token_id is [2, 24]: 2 is the real EOS,
    # 24 is "</assistant>" (end of turn in this chat template). mlx-lm's
    # tokenizer wrapper only infers the single HF eos_token (id 2) from
    # tokenizer_config.json, so without this the model never stops on 24 and
    # repeats "</assistant>" until max_tokens. Small wrapper fix, not a parser.
    gen_cfg = json.loads((Path(PACK) / "generation_config.json").read_text())
    for eid in gen_cfg.get("eos_token_id", []):
        if eid not in tok.eos_token_ids:
            tok.add_eos_token(str(eid))
    return model, tok


def main():
    global slots
    if "--slots" in sys.argv:
        i = sys.argv.index("--slots")
        slots = int(sys.argv[i + 1])
        del sys.argv[i : i + 2]
    server.load = _load
    if "--model" not in sys.argv:
        sys.argv += ["--model", PACK]
    server.main()


if __name__ == "__main__":
    main()
