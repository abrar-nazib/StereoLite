"""Tiny local sink: POST /save?name=<file>.svg with the SVG body -> written next to this script."""
import http.server, pathlib, urllib.parse

HERE = pathlib.Path(__file__).parent


class H(http.server.BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_POST(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = pathlib.Path(q.get("name", ["out.svg"])[0]).name
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        (HERE / name).write_bytes(body)
        self.send_response(200); self._cors(); self.end_headers()
        self.wfile.write(f"saved {name} {n} bytes".encode())

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", 8766), H).serve_forever()
