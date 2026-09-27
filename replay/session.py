"""Captured pool session (proxy JSONL) -> Laguna chat-template tokens + assistant mask."""
import json
from pathlib import Path

import numpy as np


def normalize(m):
    """OpenAI wire format -> what Laguna's chat template expects."""
    m = dict(m)
    c = m.get("content")
    if isinstance(c, list):
        m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    elif c is None:
        m["content"] = ""
    if m.get("tool_calls"):
        calls = []
        for tc in m["tool_calls"]:
            args = tc["function"].get("arguments", {})
            if isinstance(args, str):
                original = args
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": original}  # model emitted malformed JSON; keep the text
                # JSON parsed to non-dict (null, list, number, bool) -> wrap in _raw
                if not isinstance(args, dict):
                    args = {"_raw": original}
            calls.append({**tc, "function": {**tc["function"], "arguments": args}})
        m["tool_calls"] = calls
    return m


def load_session(path):
    """(messages, tools) of the last chat request: it holds the whole conversation so far."""
    reqs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    reqs = [r for r in reqs if r["path"].endswith("/chat/completions") and isinstance(r["request"], dict)]
    if not reqs:
        raise ValueError(f"{path}: no chat requests")
    last = reqs[-1]["request"]
    return [normalize(m) for m in last["messages"]], last.get("tools")


def tokenize(tokenizer, messages, tools, max_tokens=None):
    # mlx_lm.load() returns a TokenizerWrapper whose apply_chat_template
    # hardcodes return_dict=False before delegating to its wrapped HF
    # tokenizer, which breaks return_assistant_tokens_mask=True. Unwrap
    # only that wrapper: a plain HF (fast) tokenizer also has a `_tokenizer`
    # attribute, but it's the low-level Rust tokenizer with no
    # apply_chat_template, so it must be left alone.
    if type(tokenizer).__name__ == "TokenizerWrapper":
        tokenizer = tokenizer._tokenizer
    enc = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, return_dict=True,
        return_assistant_tokens_mask=True, enable_thinking=True,
    )
    ids = np.array(enc["input_ids"], np.int32)
    mask = np.array(enc["assistant_masks"], bool)
    truncated = max_tokens is not None and len(ids) > max_tokens
    if truncated:
        ids, mask = ids[:max_tokens], mask[:max_tokens]
    return ids, mask, truncated
