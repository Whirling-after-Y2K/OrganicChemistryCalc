"""本地 Web 分子查看器服务（仅绑定 127.0.0.1，Python 标准库实现）。

用法：
    python oc_web.py                 # 启动服务并打开浏览器
    python oc_web.py <分子文件路径>   # 启动后自动加载该文件
    python oc_web.py <端口> <文件>

接口：
    GET  /                 主页（web/index.html）
    GET  /api/health       健康检查
    POST /api/load         加载分子；JSON 请求体为 {"path": ...} 或
                           {"filename": ..., "content": ...}
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import quote, urlparse

import oc_io
import oc_render

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class MoleculeViewerHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理：静态文件 + /api/load。"""

    def log_message(self, format: str, *args: object) -> None:
        # 保留简单日志，便于排查
        sys.stderr.write("[oc_web] " + (format % args) + "\n")

    # ---- 工具 ----

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path: str) -> None:
        full = os.path.realpath(os.path.join(WEB_DIR, rel_path))
        web_root = os.path.realpath(WEB_DIR)
        if not full.startswith(web_root + os.sep) and full != web_root:
            self._send_json({"ok": False, "error": "非法路径"}, status=404)
            return
        if not os.path.isfile(full):
            self._send_json({"ok": False, "error": "文件不存在"}, status=404)
            return
        content_type = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript",
            "application/json",
        ):
            content_type += "; charset=utf-8"
        with open(full, "rb") as file:
            body = file.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- 路由 ----

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"
        if path == "/api/health":
            self._send_json({"ok": True, "service": "oc_web"})
            return
        self._send_static(path.lstrip("/"))

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/load":
            self._send_json({"ok": False, "error": "未知接口"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json(
                {"ok": False, "error": f"请求格式错误：{exc}"}, status=400
            )
            return
        try:
            if "path" in body:
                path = str(body["path"])
                molecule = oc_io.load_molecule(path)
                source = os.path.basename(path)
            elif "content" in body:
                content = str(body["content"])
                molecule = oc_io.molecule_from_code(content)
                source = str(body.get("filename") or "分子")
            else:
                raise ValueError("请求需包含 path 或 content")
            payload = oc_render.molecule_to_payload(molecule, source)
            self._send_json({"ok": True, "molecule": payload})
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json(
                {"ok": False, "error": f"加载失败：{exc}"}, status=400
            )


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    port: int = 0
    initial_file: str | None = None
    for arg in args:
        if arg.isdigit():
            port = int(arg)
        elif arg == "--no-browser":
            pass
        else:
            initial_file = arg
    if port == 0:
        port = _find_free_port()

    server = ThreadingHTTPServer(("127.0.0.1", port), MoleculeViewerHandler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print("有机分子查看器已启动，请在浏览器中打开：")
    print(url)

    open_url = url
    if initial_file:
        open_url += "?open=" + quote(initial_file)

    if "--no-browser" not in args:
        threading.Timer(0.5, lambda: webbrowser.open(open_url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
