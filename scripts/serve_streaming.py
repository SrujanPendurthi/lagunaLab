"""Serve the SSD-streaming Laguna model through mlx-lm's OpenAI-compatible server.

python3 -m scripts.serve_streaming --slots 32 --port 8081 \
    --prompt-cache-size 6 --prompt-cache-bytes 4294967296 --max-tokens 4096
python3 -m scripts.serve_streaming --preset laptop16 --port 8081   # the Part 3 answer (docs/part3-answer.md)
python3 -m scripts.serve_streaming --preset laptop16 --fused-decode   # 8-bit KV, fused decode kernel
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
from laguna_mlx import mx_without_wired_limit
from laguna_mlx.cache import make_quantized_cache
from laguna_mlx.streaming import load_streaming

PACK = "models/Laguna-XS-2.1-packed"
slots = 32
pack = PACK
cache_limit_gb = 1.0
# KV cache: plain (model.make_cache) unless --fused-decode: 8-bit K/V on the full-attention layers,
# decoded by the fused kernel (laguna_mlx.q8_attn). Its cache has no `merge`, so mlx_lm.server
# serves requests one at a time (its non-batched path) instead of through its batch generator.
fused_decode = False
PRESETS = {
    # Part 3 answer: 10.16 GB peak, 6.87 tok/s decode at 82,671 tokens (results/kv.json).
    # ponytail: the prompt-cache limits are the ones pool ran with on 24 GB; their memory on
    # a 16 GB Mac isn't measured (serve end-to-end is an open item). Explicit flags win.
    "laptop16": {"pack": "models/Laguna-XS-2.1-pruned-50-packed", "slots": 64, "cache_limit_gb": 1.0,
                 # off: plain decodes faster (6.07 vs 5.70 tok/s @82k) and is exact; fused saves
                 # 1.45 GB peak (results/kv_stream_*.json, docs/results.md)
                 "fused_decode": False,
                 "server_args": {"--max-tokens": "4096", "--prompt-cache-size": "6",
                                 "--prompt-cache-bytes": "4294967296"}},
}


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
    model, tok = load_streaming(pack, slots, cache_limit_gb=cache_limit_gb)
    if fused_decode:
        model.make_cache = lambda: make_quantized_cache(model, 8, fused_decode=True)
    # generation_config.json's eos_token_id is [2, 24]: 2 is the real EOS,
    # 24 is "</assistant>" (end of turn in this chat template). mlx-lm's
    # tokenizer wrapper only infers the single HF eos_token (id 2) from
    # tokenizer_config.json, so without this the model never stops on 24 and
    # repeats "</assistant>" until max_tokens. Small wrapper fix, not a parser.
    gen_cfg = json.loads((Path(pack) / "generation_config.json").read_text())
    for eid in gen_cfg.get("eos_token_id", []):
        if eid not in tok.eos_token_ids:
            tok.add_eos_token(str(eid))
    return model, tok


def apply_preset(argv):
    """Pop `--preset NAME` from argv, set the module config, and add its server flags that argv
    doesn't already set. Returns the resolved config."""
    global pack, slots, cache_limit_gb, fused_decode
    if "--preset" not in argv:
        return None
    i = argv.index("--preset")
    p = PRESETS[argv[i + 1]]
    del argv[i : i + 2]
    pack, slots, cache_limit_gb = p["pack"], p["slots"], p["cache_limit_gb"]
    fused_decode = p["fused_decode"]
    for flag, value in p["server_args"].items():
        if flag not in argv:
            argv += [flag, value]
    return {**p, "server_args": {f: argv[argv.index(f) + 1] for f in p["server_args"]}}


def main():
    global slots, fused_decode
    preset = apply_preset(sys.argv)
    if "--fused-decode" in sys.argv:  # explicit flag wins over the preset
        sys.argv.remove("--fused-decode")
        fused_decode = True
    if "--slots" in sys.argv:
        i = sys.argv.index("--slots")
        slots = int(sys.argv[i + 1])
        del sys.argv[i : i + 2]
    server.load = _load
    # mlx_lm.server.main raises the wired limit to the max working set; streaming doesn't
    # need it, and it pins memory the rest of the machine needs (it stalled video playback)
    server.mx = mx_without_wired_limit()
    if "--model" not in sys.argv:
        sys.argv += ["--model", pack]
    print(f"serve_streaming: pack={pack} slots={slots} cache_limit_gb={cache_limit_gb} "
          f"kv={'q8-fused' if fused_decode else 'plain'}"
          + (f" preset={preset['server_args']}" if preset else ""), flush=True)
    server.main()


if __name__ == "__main__":
    main()
