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
import uuid
import webbrowser
from typing import Any
from urllib.parse import quote


from flask import Flask, jsonify, request


import oc_io
import oc_render
import organic_chemistry as oc

app = Flask(__name__, static_folder="web", static_url_path="")
app.json.ensure_ascii = False  # 保留中文，便于调试与阅读

# 编辑会话：session_id -> {"molecule": Molecule, "source": str}（仅存内存）
_SESSIONS: dict[str, dict[str, Any]] = {}


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
        session_id = uuid.uuid4().hex
        _SESSIONS[session_id] = {"molecule": molecule, "source": source}
        return jsonify({"ok": True, "session_id": session_id, "molecule": payload})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"加载失败：{exc}"}), 400


def _get_session(session_id: str) -> dict[str, Any]:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise ValueError("会话不存在，请重新打开分子")
    return session


def _atom_at(molecule: oc.Molecule, atom_id: object) -> oc.Atom:
    try:
        index = int(atom_id)
    except (TypeError, ValueError):
        raise ValueError("原子编号无效") from None
    if not 0 <= index < len(molecule.atoms):
        raise ValueError("原子编号越界")
    return molecule.atoms[index]


def _in_any_pi(atom: oc.Atom) -> bool:
    return any(atom in pi.atoms for pi in atom.belong.pi_systems)


@app.post("/api/edit")
def edit_molecule() -> tuple[Any, int]:
    """编辑分子：{session_id, op, ...}；成功后返回最新载荷。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
    try:
        session = _get_session(str(body.get("session_id") or ""))
        molecule = session["molecule"]
        op = str(body.get("op") or "")
        if op == "add_atom":
            element = str(body.get("element") or "")
            if element not in oc.CHEMISTRY_BOND_DICT:
                raise ValueError(f"不支持的元素：{element}")
            oc.Atom(element, molecule)
        elif op == "add_atom_bonded":
            anchor = _atom_at(molecule, body.get("atom"))
            element = str(body.get("element") or "")
            if element not in oc.CHEMISTRY_BOND_DICT:
                raise ValueError(f"不支持的元素：{element}")
            order = int(body.get("order") or 1)
            new_atom = oc.Atom(element, molecule)
            try:
                oc.add_bond(anchor, new_atom, order)
            except ValueError:
                molecule.atoms.remove(new_atom)  # 回滚刚创建但未成键的原子
                raise
        elif op == "add_bond":
            atom1 = _atom_at(molecule, body.get("atom1"))
            atom2 = _atom_at(molecule, body.get("atom2"))
            order = int(body.get("order") or 1)
            if not 1 <= order <= oc.MAX_BOND_ORDER:
                raise ValueError(f"键级必须为 1-{oc.MAX_BOND_ORDER}")
            oc.add_bond(atom1, atom2, order)
        elif op == "set_bond_order":
            atom1 = _atom_at(molecule, body.get("atom1"))
            atom2 = _atom_at(molecule, body.get("atom2"))
            order = int(body.get("order") or 1)
            if not 1 <= order <= oc.MAX_BOND_ORDER:
                raise ValueError(f"键级必须为 1-{oc.MAX_BOND_ORDER}")
            if _in_any_pi(atom1) or _in_any_pi(atom2):
                raise ValueError("该键涉及 π 体系，暂不支持调整键级")
            bond = None
            for candidate in atom1.bonds:
                if candidate.other(atom1) is atom2:
                    bond = candidate
                    break
            if bond is None:
                oc.add_bond(atom1, atom2, order)
            elif order > bond.order:
                oc.add_bond(atom1, atom2, order - bond.order)
            elif order < bond.order:
                oc.break_bond(atom1, atom2, bond.order - order)
        elif op == "del_atom":
            atom = _atom_at(molecule, body.get("atom"))
            if _in_any_pi(atom):
                raise ValueError("该原子参与 π 体系，暂不支持删除")
            oc.del_atom(atom)
        elif op == "del_bond":
            atom1 = _atom_at(molecule, body.get("atom1"))
            atom2 = _atom_at(molecule, body.get("atom2"))
            if _in_any_pi(atom1) or _in_any_pi(atom2):
                raise ValueError("该键涉及 π 体系，暂不支持删除")
            oc.break_bond(atom1, atom2)
        else:
            raise ValueError("未知编辑操作")
        payload = oc_render.molecule_to_payload(molecule, session["source"])
        return jsonify({"ok": True, "session_id": body["session_id"], "molecule": payload})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"编辑失败：{exc}"}), 400


@app.post("/api/save")
def save_molecule() -> tuple[Any, int]:
    """保存分子为本地构建脚本：{session_id, path}。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
    try:
        session = _get_session(str(body.get("session_id") or ""))
        path = str(body.get("path") or "").strip()
        if not path:
            raise ValueError("保存路径不能为空")
        oc_io.save_molecule(session["molecule"], path)
        return jsonify({"ok": True, "path": os.path.abspath(path)})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"保存失败：{exc}"}), 400


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
