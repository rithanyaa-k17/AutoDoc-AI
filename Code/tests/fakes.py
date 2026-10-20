"""Test doubles (offline, no model download, no credentials).
FakeEmbedder is a hashed bag-of-words encoder: it exercises the embedding/vector-store PLUMBING; it is NOT
semantic and is never used by the application. MockLLMServer imitates an OpenAI-compatible endpoint."""
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class FakeEmbedder:
    name = "fake-hash-embedder"
    dim = 256

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in ENGLISH_STOP_WORDS:
                    continue
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class MockLLMServer:
    """reply: callable(messages)->str  or str. Records every request (so tests can assert on prompts/calls)."""

    def __init__(self, reply="Answer [S1]."):
        self.reply, self.requests = reply, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": "mock-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                req["_auth"] = self.headers.get("Authorization")
                outer.requests.append(req)
                text = outer.reply(req["messages"]) if callable(outer.reply) else outer.reply
                body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
