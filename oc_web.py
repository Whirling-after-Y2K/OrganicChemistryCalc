"""本地 Web 分子查看器服务（Flask，仅绑定 127.0.0.1）。

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

import os
import sys
import threading
import webbrowser
from typing import Any
from urllib.parse import quote


from flask import Flask, jsonify, request


import oc_io
import oc_render

app = Flask(__name__, static_folder="web", static_url_path="")
app.json.ensure_ascii = False  # 保留中文，便于调试与阅读


@app.get("/")
def index() -> Any:
    """主页。"""
    return app.send_static_file("index.html")


@app.get("/api/health")
def health() -> Any:
    """健康检查。"""
    return jsonify({"ok": True, "service": "oc_web"})


@app.post("/api/load")
def load_molecule() -> tuple[Any, int]:
    """加载分子：接受 {"path": ...} 或 {"filename": ..., "content": ...}。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
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
        return jsonify({"ok": True, "molecule": payload})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"加载失败：{exc}"}), 400


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

    open_url = f"http://127.0.0.1:{port}/"
    print("有机分子查看器已启动，请在浏览器中打开：")
    print(open_url)
    if initial_file:
        open_url += "?open=" + quote(initial_file)

    if "--no-browser" not in args:
        threading.Timer(0.5, lambda: webbrowser.open(open_url)).start()
    try:
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
