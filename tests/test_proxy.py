import http.server
import json
import threading
import urllib.error
import urllib.request

from scripts.proxy import make_handler


class Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(500 if self.path == "/fail" else 200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for part in (b'data: {"a":1}\n\n', b"data: [DONE]\n\n"):
            self.wfile.write(part)
            self.wfile.flush()

    def log_message(self, *a):
        pass


def serve(handler):
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def post(url, body):
    return urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}))


def test_forwards_stream_logs_requests_and_errors(tmp_path):
    up = serve(Upstream)
    log = tmp_path / "t.jsonl"
    px = serve(make_handler(f"http://127.0.0.1:{up.server_port}", str(log)))
    base = f"http://127.0.0.1:{px.server_port}"
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()

    assert post(base + "/v1/chat/completions", body).read() == b'data: {"a":1}\n\ndata: [DONE]\n\n'
    try:
        post(base + "/fail", body)
        raise AssertionError("expected HTTPError")
    except urllib.error.HTTPError as e:
        assert e.code == 500

    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["status"] for l in lines] == [200, 500]
    assert lines[0]["request"]["messages"][0]["content"] == "hi"
    assert "[DONE]" in lines[0]["response"]


def test_unreachable_upstream_returns_502_and_logs(tmp_path):
    log = tmp_path / "t.jsonl"
    px = serve(make_handler("http://127.0.0.1:9", str(log)))  # port 9: nothing listening
    try:
        post(f"http://127.0.0.1:{px.server_port}/v1/chat/completions", b"{}")
        raise AssertionError("expected HTTPError")
    except urllib.error.HTTPError as e:
        assert e.code == 502
    assert json.loads(log.read_text())["status"] == 502
