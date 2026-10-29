import json
import os
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import agent
import pipeline
import tools

ROOT = Path(__file__).resolve().parent
registry = pipeline.StoreRegistry(ROOT / "data")


def _json(start_response, payload, status="200 OK"):
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    start_response(status, [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))])
    return [body]


def _body(environ):
    try:
        size = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        size = 0
    raw = environ["wsgi.input"].read(min(size, 2_000_000)) if size else b""
    return json.loads(raw.decode("utf-8") or "{}")


def _context(payload):
    state = tools.sanitize_state(payload.get("state"), registry.names())
    if payload.get("store") in registry.names():
        state["store"] = payload["store"]
    return tools.ToolContext(registry, state)


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")
    try:
        if path == "/" and method == "GET":
            body = (ROOT / "static" / "index.html").read_bytes()
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]
        if path == "/api/stores":
            return _json(start_response, {"stores": registry.names()})
        if path == "/api/summary":
            store = registry.get(parse_qs(environ.get("QUERY_STRING", "")).get("store", [""])[0])
            return _json(start_response, {"metrics": store.metrics, "quality": store.quality,
                                          "daily": tools.records(store.table("daily_revenue"))})
        if path == "/api/table" and method == "POST":
            payload = _body(environ)
            ctx = _context(payload)
            return _json(start_response, tools.table_page(ctx, payload.get("table"), payload.get("page", 1), payload.get("page_size", 25)))
        if path == "/api/chat" and method == "POST":
            payload = _body(environ)
            message = str(payload.get("message", "")).strip()[:4000]
            if not message:
                return _json(start_response, {"error": "empty message"}, "400 Bad Request")
            return _json(start_response, agent.chat(registry, payload.get("state"), payload.get("history"), message))
        return _json(start_response, {"error": "not found"}, "404 Not Found")
    except (KeyError, tools.ToolError, ValueError) as e:
        return _json(start_response, {"error": str(e.args[0] if e.args else e)}, "400 Bad Request")
    except Exception as e:
        traceback.print_exc()
        return _json(start_response, {"error": f"{type(e).__name__}: {e}"}, "500 Internal Server Error")


class _Server(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _QuietHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if not str(code).startswith("2"):
            super().log_request(code, size)


def _warm_up(names):
    for name in names:
        registry.get(name)
        print(f"{name} loaded", flush=True)


def main():
    sys.stdout.reconfigure(errors="replace")
    port = int(os.environ.get("PORT", "8000"))
    names = registry.names()
    if not names:
        print(f"No stores in {ROOT / 'data'}. Copy STORE1 there (orders.parquet, users.csv, books.yaml).")
        return 1
    threading.Thread(target=_warm_up, args=(names,), daemon=True).start()
    if not agent.config()["api_key"]:
        print("Warning: LLM_API_KEY is not set, the chat will not answer (see .env.example).")
    url = f"http://localhost:{port}"
    print(f"Dashboard: {url}  (Ctrl+C to stop)", flush=True)
    if os.environ.get("OPEN_BROWSER", "0" if os.environ.get("RENDER") else "1") == "1":
        threading.Timer(1.0, webbrowser.open, [url]).start()
    with make_server(os.environ.get("HOST", "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"), port, application, server_class=_Server, handler_class=_QuietHandler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
