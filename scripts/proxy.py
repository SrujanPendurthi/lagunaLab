"""Logging reverse proxy for an OpenAI-compatible server (streaming preserved).

python3 scripts/proxy.py --listen 11434 --upstream http://127.0.0.1:11435 --out traces/x.jsonl
"""
import argparse
import http.server
import json
import urllib.error
import urllib.request

HOP = {"host", "content-length", "transfer-encoding", "connection"}


class _ClientWriter:
    """Wraps the handler's wfile: a vanished client ends output, never the request."""

    def __init__(self, raw, handler):
        self.raw, self.handler = raw, handler

    def write(self, data):
        if not self.handler.client_gone:
            try:
                self.raw.write(data)
            except (BrokenPipeError, ConnectionResetError):
                self.handler.client_gone = True
                self.handler.close_connection = True
        return len(data)

    def flush(self):
        if not self.handler.client_gone:
            try:
                self.raw.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.handler.client_gone = True
                self.handler.close_connection = True

    def close(self):
        try:
            self.raw.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    @property
    def closed(self):
        return self.raw.closed


def make_handler(upstream, out):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        client_gone = False

        def setup(self):
            super().setup()
            self.client_gone = False
            self.wfile = _ClientWriter(self.wfile, self)

        def _relay(self, body):
            """Forward, stream the body back as chunks, return (status, raw bytes). No terminator."""
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            req = urllib.request.Request(upstream + self.path, data=body, method=self.command, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=600)
            except urllib.error.HTTPError as e:
                resp = e
            except TimeoutError:
                msg = b"proxy: upstream timeout"
                self.send_response(504)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(msg), msg))
                return 504, msg
            except urllib.error.URLError as e:
                msg = f"proxy: upstream unreachable: {e.reason}".encode()
                self.send_response(502)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(msg), msg))
                return 502, msg
            try:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                read = getattr(resp, "read1", resp.read)
                chunks = []
                while True:
                    try:
                        chunk = read(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    except TimeoutError:
                        # Timeout during streaming, return what we have so far
                        break
                return resp.status, b"".join(chunks)
            finally:
                if hasattr(resp, 'close'):
                    resp.close()

        def _finish(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                status, raw = self._relay(body)
                try:
                    request = json.loads(body)
                except ValueError:
                    request = body.decode(errors="replace")
                with open(out, "a") as f:  # log before finishing, so a reader never races the write
                    f.write(json.dumps({"path": self.path, "request": request, "status": status,
                                        "response": raw.decode(errors="replace")}) + "\n")
            finally:
                self._finish()

        def do_GET(self):
            self._relay(None)
            self._finish()

        def log_message(self, *a):
            pass

    return Handler


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    http.server.ThreadingHTTPServer(("127.0.0.1", a.listen), make_handler(a.upstream, a.out)).serve_forever()
