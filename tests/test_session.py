import json

import pytest
from transformers import AutoTokenizer

from replay.session import load_session, normalize, tokenize

TOK = "models/Laguna-XS-2.1"
TOOL_CALL = {"id": "c1", "type": "function",
             "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": [{"type": "text", "text": "list files"}]},
    {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.py\nb.py"},
    {"role": "assistant", "content": "There are two files."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOK)


def test_normalize_handles_openai_shapes():
    assert normalize(MESSAGES[1])["content"] == "list files"
    a = normalize(MESSAGES[2])
    assert a["content"] == "" and a["tool_calls"][0]["function"]["arguments"] == {"cmd": "ls"}
    bad = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": "{not json"}}]}
    assert normalize(bad)["tool_calls"][0]["function"]["arguments"] == {"_raw": "{not json"}
    # JSON-parseable non-dict values should also be wrapped in _raw
    null_args = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": "null"}}]}
    assert normalize(null_args)["tool_calls"][0]["function"]["arguments"] == {"_raw": "null"}
    list_args = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": "[]"}}]}
    assert normalize(list_args)["tool_calls"][0]["function"]["arguments"] == {"_raw": "[]"}


def test_load_session_uses_last_chat_request(tmp_path):
    f = tmp_path / "s.jsonl"
    lines = [
        {"path": "/v1/chat/completions", "request": {"messages": MESSAGES[:2]}, "status": 200, "response": ""},
        {"path": "/v1/chat/completions", "request": {"messages": MESSAGES, "tools": [{"x": 1}]}, "status": 200, "response": ""},
        {"path": "/api/show", "request": "not json", "status": 200, "response": ""},
    ]
    f.write_text("\n".join(json.dumps(l) for l in lines))
    msgs, tools = load_session(f)
    assert len(msgs) == 5 and tools == [{"x": 1}]


def test_mask_covers_assistant_turns_only(tok):
    msgs = [normalize(m) for m in MESSAGES]
    ids, mask, truncated = tokenize(tok, msgs, None)
    assert not truncated and len(ids) == len(mask)
    inside, outside = tok.decode(ids[mask]), tok.decode(ids[~mask])
    assert "<tool_call>bash" in inside and "There are two files." in inside
    assert "list files" in outside and "a.py" in outside and "list files" not in inside


def test_truncation_is_flagged(tok):
    msgs = [normalize(m) for m in MESSAGES]
    ids, mask, truncated = tokenize(tok, msgs, None, max_tokens=10)
    assert truncated and len(ids) == 10 and len(mask) == 10


def test_tokenize_with_null_arguments(tok):
    # Tool call with "null" arguments (parsed JSON non-dict) should tokenize without error
    msgs = [
        normalize({"role": "system", "content": "You are a helper."}),
        normalize({"role": "user", "content": "do something"}),
        normalize({"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "no_args", "arguments": "null"}}
        ]}),
    ]
    ids, mask, truncated = tokenize(tok, msgs, None)
    assert len(ids) == len(mask) and not truncated
