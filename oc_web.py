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
    POST /api/templates/<template>
                           新建内置模板分子：benzene / nitro
    POST /api/isomers/jobs 后台枚举同分异构体，返回 {"job_id": ...}
    POST /api/synthesis/jobs 后台规划合成路线，返回 {"job_id": ...}
    GET  /api/analysis-jobs/<job_id> 查询后台分析任务状态
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote


from flask import Flask, jsonify, request


import oc_io
import oc_isomers
import oc_render
import oc_synthesis
import organic_chemistry as oc

app = Flask(__name__, static_folder="web", static_url_path="")
app.json.ensure_ascii = False  # type: ignore # 保留中文，便于调试与阅读

# 编辑会话：session_id -> {"molecule": Molecule, "source": str}（仅存内存）
_SESSIONS: dict[str, dict[str, Any]] = {}

# 已解析分子的载荷缓存：路径按 mtime/size 失效，内容按 SHA-256 失效。
# 会话始终使用深拷贝，避免编辑操作污染缓存。
_LOAD_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
MAX_LOAD_CACHE_ENTRIES: int = 256
SLOW_LOAD_WARNING_MS: float = 100.0

# 一次接口返回的异构体上限，避免超大枚举把浏览器响应压垮
MAX_ISOMER_RESULTS: int = 500

# 后台分析任务：提交后立即返回 job_id，前端轮询状态，避免长耗时搜索阻塞页面。
_ANALYSIS_JOBS: dict[str, dict[str, Any]] = {}
MAX_ANALYSIS_JOBS: int = 64
ANALYSIS_POLL_INTERVAL_MS: int = 250


@app.get("/")
def index() -> Any:
    """主页。"""
    return app.send_static_file("index.html")


@app.get("/api/health")
def health() -> Any:
    """健康检查。"""
    return jsonify({"ok": True, "service": "oc_web"})


@app.get("/api/groups")
def groups() -> Any:
    """返回同分异构体分析可用的官能团约束。"""
    return jsonify({"ok": True, "groups": list(oc_isomers.REQUIRED_GROUP_NAMES)})


def _path_cache_key(path: str, source: str) -> tuple[Any, ...]:
    """路径加载缓存键；文件修改时间或大小变化后自动失效。"""
    absolute = os.path.normcase(os.path.abspath(path))
    stat = os.stat(path)
    return ("path", absolute, stat.st_mtime_ns, stat.st_size, source)


def _content_cache_key(content: str, source: str) -> tuple[Any, ...]:
    """内容加载缓存键，同一构建内容重复上传时不再执行代码。"""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return ("content", digest, source)


def _cache_loaded_molecule(
    key: tuple[Any, ...],
    molecule: oc.Molecule,
    payload: dict[str, Any],
) -> None:
    """保存解析结果；简单 FIFO 上限防止长期运行时缓存无限增长。"""
    if key not in _LOAD_CACHE and len(_LOAD_CACHE) >= MAX_LOAD_CACHE_ENTRIES:
        oldest = next(iter(_LOAD_CACHE))
        del _LOAD_CACHE[oldest]
    _LOAD_CACHE[key] = {"molecule": molecule, "payload": payload}


@app.post("/api/load")
def load_molecule() -> Any:
    """加载分子：接受 {"path": ...} 或 {"filename": ..., "content": ...}。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
    try:
        request_start = time.perf_counter()
        parse_ms: float = 0.0
        render_ms: float = 0.0

        if "path" in body:
            path = str(body["path"])
            source = os.path.basename(path)
            cache_key = _path_cache_key(path, source)
        elif "content" in body:
            content = str(body["content"])
            source = str(body.get("filename") or "分子")
            cache_key = _content_cache_key(content, source)
        else:
            raise ValueError("请求需包含 path 或 content")

        cached = _LOAD_CACHE.get(cache_key)
        if cached is None:
            parse_start = time.perf_counter()
            if "path" in body:
                cached_molecule = oc_io.load_molecule(str(body["path"]))
            else:
                cached_molecule = oc_io.molecule_from_code(str(body["content"]))
            parse_ms = round((time.perf_counter() - parse_start) * 1000, 3)

            render_start = time.perf_counter()
            payload = oc_render.molecule_to_payload(cached_molecule, source)
            render_ms = round((time.perf_counter() - render_start) * 1000, 3)
            _cache_loaded_molecule(cache_key, cached_molecule, payload)
        else:
            cached_molecule = cached["molecule"]
            payload = cached["payload"]

        session_id = uuid.uuid4().hex
        copy_start = time.perf_counter()
        session_molecule = oc.copy_molecule(cached_molecule)
        copy_ms = round((time.perf_counter() - copy_start) * 1000, 3)
        _SESSIONS[session_id] = {
            "molecule": session_molecule,
            "source": source,
        }
        total_ms = round((time.perf_counter() - request_start) * 1000, 3)
        timing: dict[str, Any] = {
            "cache_hit": cached is not None,
            "parse_ms": parse_ms,
            "render_ms": render_ms,
            "copy_ms": copy_ms,
            "total_ms": total_ms,
        }
        if total_ms > SLOW_LOAD_WARNING_MS:
            app.logger.warning(
                "分子加载耗时 %.3f ms 超过 %.3f ms：source=%s, timing=%s",
                total_ms,
                SLOW_LOAD_WARNING_MS,
                source,
                timing,
            )
        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
                "molecule": payload,
                "timing": timing,
            }
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"加载失败：{exc}"}), 400


def _get_session(session_id: str) -> dict[str, Any]:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise ValueError("会话不存在，请重新打开分子")
    return session


def _create_benzene() -> oc.Molecule:
    """创建苯模板；苯环使用单键骨架加离域 π 体系表示。"""
    molecule = oc.Molecule(name="苯")
    ring = [oc.Atom("c", molecule) for _ in range(6)]
    for index, atom in enumerate(ring):
        oc.add_bond(atom, ring[(index + 1) % 6])
    oc.add_pi_system(ring)
    molecule.validate()
    return molecule


def _create_nitro_methane() -> oc.Molecule:
    """创建硝基模板；硝基以 [N,O,O] π 体系连接到甲基。"""
    molecule = oc.Molecule(name="硝基甲烷")
    carbon = oc.Atom("c", molecule)
    nitrogen = oc.Atom("n", molecule)
    oxygen1 = oc.Atom("o", molecule)
    oxygen2 = oc.Atom("o", molecule)
    oc.add_bond(carbon, nitrogen)
    oc.add_bond(nitrogen, oxygen1)
    oc.add_bond(nitrogen, oxygen2)
    oc.add_pi_system([nitrogen, oxygen1, oxygen2])
    molecule.validate()
    return molecule


_TEMPLATE_CREATORS: dict[str, tuple[str, Callable[[], oc.Molecule]]] = {
    "benzene": ("苯", _create_benzene),
    "nitro": ("硝基甲烷", _create_nitro_methane),
}


@app.post("/api/templates/<template>")
def create_template_molecule(template: str) -> Any:
    """新建内置模板分子并返回编辑会话。"""
    item = _TEMPLATE_CREATORS.get(template)
    if item is None:
        return jsonify({"ok": False, "error": "未知分子模板"}), 400
    try:
        source, create_molecule = item
        molecule = create_molecule()
        session_id = uuid.uuid4().hex
        _SESSIONS[session_id] = {
            "molecule": molecule,
            "source": source,
        }
        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
                "molecule": oc_render.molecule_to_payload(molecule, source),
            }
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"新建模板失败：{exc}"}), 400


def _atom_at(molecule: oc.Molecule, atom_id: object) -> oc.Atom:
    try:
        index = int(atom_id) #type: ignore
    except (TypeError, ValueError):
        raise ValueError("原子编号无效") from None
    if not 0 <= index < len(molecule.atoms):
        raise ValueError("原子编号越界")
    return molecule.atoms[index]


def _add_benzene_to_atom(molecule: oc.Molecule, anchor: oc.Atom) -> None:
    """在锚点原子上接入一个苯环取代基。"""
    ring = [oc.Atom("c", molecule) for _ in range(6)]
    for index, atom in enumerate(ring):
        oc.add_bond(atom, ring[(index + 1) % 6])
    oc.add_pi_system(ring)
    oc.add_bond(anchor, ring[0])
    molecule.validate()


def _add_nitro_to_atom(molecule: oc.Molecule, anchor: oc.Atom) -> None:
    """在碳锚点上接入一个硝基取代基。"""
    if anchor.name != "c":
        raise ValueError("硝基只能连接到碳原子")
    nitrogen = oc.Atom("n", molecule)
    oxygen1 = oc.Atom("o", molecule)
    oxygen2 = oc.Atom("o", molecule)
    oc.add_bond(anchor, nitrogen)
    oc.add_bond(nitrogen, oxygen1)
    oc.add_bond(nitrogen, oxygen2)
    oc.add_pi_system([nitrogen, oxygen1, oxygen2])
    molecule.validate()


def _in_any_pi(atom: oc.Atom) -> bool:
    return any(atom in pi.atoms for pi in atom.belong.pi_systems)


def _formula_text(formula: dict[str, int]) -> str:
    """把公式字典转成 UI 使用的 Hill 式文本。"""
    return "".join(
        element.capitalize() + (str(count) if count != 1 else "")
        for element, count in formula.items()
        if count > 0
    )


def _register_molecule(molecule: oc.Molecule, source: str) -> str:
    """把分析结果注册成普通编辑会话，供前端直接打开。"""
    session_id = uuid.uuid4().hex
    _SESSIONS[session_id] = {"molecule": molecule, "source": source}
    return session_id


def _molecule_payload(
    molecule: oc.Molecule,
    source: str,
) -> dict[str, Any]:
    """生成带会话编号的完整前端载荷。"""
    session_id = _register_molecule(molecule, source)
    return {
        "session_id": session_id,
        "molecule": oc_render.molecule_to_payload(molecule, source),
    }


def _molecule_summary(molecule: oc.Molecule) -> dict[str, str]:
    """用于路线列表的轻量分子摘要。"""
    return {
        "name": molecule.name or "",
        "formula": _formula_text(molecule.formula),
    }


def _parse_positive_int(
    body: dict[str, Any],
    name: str,
    default: int,
    maximum: int,
) -> int:
    value = body.get(name, default)
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须为整数") from None
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} 必须为 1-{maximum}")
    return result


def _isomer_analysis_result(
    molecule: oc.Molecule,
    required_groups: list[str],
    equivalent_hydrogens: list[int] | None,
    allow_extra_rings: bool | None,
    limit: int,
) -> dict[str, Any]:
    """执行异构体枚举并序列化为前端载荷（隐藏输入分子本身）。"""
    isomers = oc_isomers.find_isomers(
        molecule,
        required_groups=required_groups,
        equivalent_hydrogens=equivalent_hydrogens,
        allow_extra_rings=allow_extra_rings,
    )
    visible_isomers = [candidate for candidate in isomers if candidate != molecule]
    returned = visible_isomers[:limit]
    formula = _formula_text(molecule.formula)
    items: list[dict[str, Any]] = []
    for index, candidate in enumerate(returned, 1):
        source = f"{formula} 异构体 {index:02d}"
        item = _molecule_payload(candidate, source)
        item.update(
            {
                "index": index,
                "formula": _formula_text(candidate.formula),
                "unsaturation": candidate.unsaturation,
                "ring_count": candidate.ring_count,
            }
        )
        items.append(item)
    return {
        "ok": True,
        "total": len(visible_isomers),
        "returned": len(items),
        "truncated": len(visible_isomers) > len(items),
        "isomers": items,
    }


def _parse_isomer_request(
    body: dict[str, Any],
) -> tuple[
    oc.Molecule,
    list[str],
    list[int] | None,
    bool | None,
    int,
]:
    """解析并快照异构体分析请求，避免后台任务受到后续编辑影响。"""
    session = _get_session(str(body.get("session_id") or ""))
    molecule = oc.copy_molecule(session["molecule"])

    raw_groups = body.get("required_groups") or []
    if isinstance(raw_groups, str):
        required_groups = [name.strip() for name in raw_groups.split(",") if name.strip()]
    elif isinstance(raw_groups, list):
        required_groups = [str(name).strip() for name in raw_groups if str(name).strip()]
    else:
        raise ValueError("required_groups 必须为字符串数组")

    raw_hydrogens = body.get("equivalent_hydrogens")
    equivalent_hydrogens: list[int] | None
    if raw_hydrogens is None or raw_hydrogens == "":
        equivalent_hydrogens = None
    elif isinstance(raw_hydrogens, str):
        tokens = [token.strip() for token in raw_hydrogens.split(",")]
        equivalent_hydrogens = [int(token) for token in tokens if token] or None
    elif isinstance(raw_hydrogens, list):
        equivalent_hydrogens = [int(value) for value in raw_hydrogens] or None
    else:
        raise ValueError("equivalent_hydrogens 必须为正整数数组")

    limit = _parse_positive_int(body, "limit", 200, MAX_ISOMER_RESULTS)
    raw_allow_extra_rings = body.get("allow_extra_rings")
    allow_extra_rings = (
        None if raw_allow_extra_rings is None else bool(raw_allow_extra_rings)
    )
    return molecule, required_groups, equivalent_hydrogens, allow_extra_rings, limit


def _synthesis_analysis_result(
    reactants: list[oc.Molecule],
    target: oc.Molecule,
    reaction: str | None,
    conditions: str | None,
    category: str | None,
    max_steps: int,
    max_routes: int,
) -> dict[str, Any]:
    """执行合成路线规划并序列化为前端载荷。"""
    routes = oc_synthesis.plan_synthesis(
        reactants,
        target,
        reaction=reaction,
        conditions=conditions,
        category=category,
        max_steps=max_steps,
        max_routes=max_routes,
    )
    serialized: list[dict[str, Any]] = []
    for route_index, route in enumerate(routes, 1):
        route_source = f"合成路线 {route_index:02d}"
        steps: list[dict[str, Any]] = []
        for step in route.steps:
            steps.append(
                {
                    "step_no": step.step_no,
                    "rule_name": step.rule_name,
                    "category": step.category,
                    "conditions": step.conditions or "",
                    "inputs": [
                        _molecule_payload(molecule, route_source)
                        for molecule in step.inputs
                    ],
                    "outputs": [
                        _molecule_payload(molecule, route_source)
                        for molecule in step.outputs
                    ],
                }
            )
        serialized.append(
            {
                "index": route_index,
                "step_count": route.step_count,
                "target": _molecule_summary(route.target),
                "steps": steps,
                "report": oc_synthesis.synthesis_report(route),
            }
        )
    return {
        "ok": True,
        "routes": serialized,
        "route_count": len(serialized),
        "target": _molecule_summary(target),
    }


def _parse_synthesis_request(
    body: dict[str, Any],
) -> tuple[
    list[oc.Molecule],
    oc.Molecule,
    str | None,
    str | None,
    str | None,
    int,
    int,
]:
    """解析并快照合成规划请求。"""
    raw_reactants = body.get("reactant_ids")
    if not isinstance(raw_reactants, list) or not raw_reactants:
        raise ValueError("请至少选择一个起始反应物")
    reactants = [
        oc.copy_molecule(_get_session(str(session_id))["molecule"])
        for session_id in raw_reactants
        if str(session_id).strip()
    ]
    if not reactants:
        raise ValueError("请至少选择一个起始反应物")
    target = oc.copy_molecule(_get_session(str(body.get("target_id") or ""))["molecule"])
    return (
        reactants,
        target,
        str(body.get("reaction") or "").strip() or None,
        str(body.get("conditions") or "").strip() or None,
        str(body.get("category") or "").strip() or None,
        _parse_positive_int(body, "max_steps", oc_synthesis.DEFAULT_MAX_STEPS, 8),
        _parse_positive_int(body, "max_routes", 5, 20),
    )


def _cleanup_analysis_jobs() -> None:
    """清理旧的已完成任务，保留最近任务供前端刷新状态。"""
    finished = [
        job_id
        for job_id, job in _ANALYSIS_JOBS.items()
        if job["status"] in {"done", "failed"}
    ]
    while len(_ANALYSIS_JOBS) > MAX_ANALYSIS_JOBS and finished:
        del _ANALYSIS_JOBS[finished.pop(0)]


def _start_analysis_job(
    kind: str,
    work: Callable[[], dict[str, Any]],
) -> str:
    """启动后台分析任务并立即返回任务编号。"""
    _cleanup_analysis_jobs()
    job_id = uuid.uuid4().hex
    job: dict[str, Any] = {
        "kind": kind,
        "status": "running",
        "started_at": time.time(),
        "elapsed_ms": 0.0,
        "result": None,
        "error": "",
    }
    _ANALYSIS_JOBS[job_id] = job

    def run_job() -> None:
        started = time.perf_counter()
        try:
            result = work()
            job["result"] = result
            job["status"] = "done"
        except ValueError as exc:
            job["error"] = str(exc)
            job["status"] = "failed"
        except Exception as exc:
            job["error"] = f"{kind}失败：{exc}"
            job["status"] = "failed"
        finally:
            job["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)

    threading.Thread(target=run_job, name=f"oc-analysis-{job_id}", daemon=True).start()
    return job_id


def _analysis_job_response(job_id: str) -> dict[str, Any]:
    job = _ANALYSIS_JOBS.get(job_id)
    if job is None:
        raise ValueError("分析任务不存在或已过期")
    response: dict[str, Any] = {
        "ok": True,
        "job_id": job_id,
        "kind": job["kind"],
        "status": job["status"],
        "elapsed_ms": (
            job["elapsed_ms"]
            if job["status"] in {"done", "failed"}
            else round((time.time() - job["started_at"]) * 1000, 3)
        ),
    }
    if job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]
    return response


@app.post("/api/isomers")
def analyze_isomers() -> Any:
    """枚举当前分子的同分异构体：{session_id, required_groups, ...}。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
    try:
        molecule, groups, hydrogens, extra_rings, limit = _parse_isomer_request(body)
        return jsonify(
            _isomer_analysis_result(
                molecule, groups, hydrogens, extra_rings, limit
            )
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"同分异构体分析失败：{exc}"}), 400


@app.post("/api/isomers/jobs")
def start_isomer_job() -> Any:
    """提交后台异构体枚举任务，立即返回任务编号。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
    try:
        molecule, groups, hydrogens, extra_rings, limit = _parse_isomer_request(body)
        job_id = _start_analysis_job(
            "isomers",
            lambda: _isomer_analysis_result(
                molecule, groups, hydrogens, extra_rings, limit
            ),
        )
        return jsonify({"ok": True, "job_id": job_id, "poll_interval_ms": ANALYSIS_POLL_INTERVAL_MS})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"提交异构体分析失败：{exc}"}), 400


@app.get("/api/analysis-jobs/<job_id>")
def analysis_job(job_id: str) -> Any:
    """查询后台分析任务状态。"""
    try:
        return jsonify(_analysis_job_response(job_id))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404


@app.post("/api/edit")
def edit_molecule() -> Any:
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
        elif op in {"add_benzene", "add_nitro"}:
            _atom_at(molecule, body.get("atom"))
            # 在副本上构建，全部校验通过后再替换会话分子。
            edited_molecule = oc.copy_molecule(molecule)
            edited_anchor = _atom_at(edited_molecule, body.get("atom"))
            if op == "add_benzene":
                _add_benzene_to_atom(edited_molecule, edited_anchor)
            else:
                _add_nitro_to_atom(edited_molecule, edited_anchor)
            session["molecule"] = edited_molecule
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
            # 在副本上删除：只删除被点击的 π 成员原子，并移除其 π 体系。
            edited_molecule = oc.copy_molecule(molecule)
            edited_atom = _atom_at(edited_molecule, body.get("atom"))
            for pi in list(edited_atom.belong.pi_systems):
                if edited_atom in pi.atoms:
                    oc.remove_pi_system(pi)
            oc.del_atom(edited_atom)
            session["molecule"] = edited_molecule
        elif op == "del_bond":
            atom1 = _atom_at(molecule, body.get("atom1"))
            atom2 = _atom_at(molecule, body.get("atom2"))
            if _in_any_pi(atom1) or _in_any_pi(atom2):
                raise ValueError("该键涉及 π 体系，暂不支持删除")
            oc.break_bond(atom1, atom2)
        else:
            raise ValueError("未知编辑操作")
        payload = oc_render.molecule_to_payload(
            session["molecule"], session["source"]
        )
        return jsonify({"ok": True, "session_id": body["session_id"], "molecule": payload}) # type: ignore
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"编辑失败：{exc}"}), 400


@app.post("/api/save")
def save_molecule() -> Any:
    """保存分子为本地构建脚本：{session_id, path}。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
    try:
        session = _get_session(str(body.get("session_id") or ""))
        path = str(body.get("path") or "").strip()
        if not path:
            raise ValueError("保存路径不能为空")
        molecule_name = Path(path).stem
        if not molecule_name:
            raise ValueError("保存文件名不能为空")
        session["molecule"].name = molecule_name
        oc_io.save_molecule(session["molecule"], path)
        return jsonify({"ok": True, "path": os.path.abspath(path)})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"保存失败：{exc}"}), 400


@app.post("/api/synthesis")
def plan_route() -> Any:
    """规划合成路线：{reactant_ids, target_id, ...}。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
    try:
        reactants, target, reaction, conditions, category, max_steps, max_routes = (
            _parse_synthesis_request(body)
        )
        return jsonify(
            _synthesis_analysis_result(
                reactants,
                target,
                reaction,
                conditions,
                category,
                max_steps,
                max_routes,
            )
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"合成路线规划失败：{exc}"}), 400


@app.post("/api/synthesis/jobs")
def start_synthesis_job() -> Any:
    """提交后台合成规划任务，立即返回任务编号。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "请求需为 JSON 对象"}), 400
    try:
        reactants, target, reaction, conditions, category, max_steps, max_routes = (
            _parse_synthesis_request(body)
        )
        job_id = _start_analysis_job(
            "synthesis",
            lambda: _synthesis_analysis_result(
                reactants,
                target,
                reaction,
                conditions,
                category,
                max_steps,
                max_routes,
            ),
        )
        return jsonify({"ok": True, "job_id": job_id, "poll_interval_ms": ANALYSIS_POLL_INTERVAL_MS}) 
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"提交合成规划失败：{exc}"}), 400


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
