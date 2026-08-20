"""分子 2D 布局与 JSON 载荷（轻量自研，无第三方依赖）。

职责：
- compute_coordinates()：把 Molecule 图布局到平面（确定性算法，多次调用结果一致）；
- molecule_to_payload()：生成前端渲染所需的结构数据（坐标、标签、键、π 体系、
  分子式、官能团等派生信息）。

显示约定（纯显示层，不修改数据模型）：
- 苯环：正六边形 + 内圈圆（成员键仍按单键输出）；
- 硝基：N-O 中一根按双键显示（取 π 体系成员顺序中第一个 O），其余为单键。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import oc_features
import organic_chemistry as oc

BOND_LEN: float = 40.0          # 布局单位下的键长
_MAX_RING_SIZE: int = 8         # 环检测上限（高中范围：3-8 元环）
_SP2_ANGLE: float = math.radians(120.0)    # sp2 理想键角
_SP3_ANGLE: float = math.radians(109.5)    # sp3 理想键角
_SUBSCRIPTS: str = "₀₁₂₃₄₅₆₇₈₉"


# ------- 布局 -------


def _adjacency(molecule: oc.Molecule) -> dict[oc.Atom, list[oc.Atom]]:
    """原子 -> 邻居列表（按 molecule.bonds 顺序）。"""
    result: dict[oc.Atom, list[oc.Atom]] = {}
    for atom in molecule.atoms:
        result[atom] = []
    for bond in molecule.bonds:
        a1, a2 = bond.atoms
        result[a1].append(a2)
        result[a2].append(a1)
    return result


def _connected_components(
    molecule: oc.Molecule, adj: dict[oc.Atom, list[oc.Atom]]
) -> list[list[oc.Atom]]:
    """连通分量：按分子内原子顺序确定分量顺序与分量内原子顺序。"""
    index: dict[oc.Atom, int] = {a: i for i, a in enumerate(molecule.atoms)}
    seen: set[oc.Atom] = set()
    components: list[list[oc.Atom]] = []
    for start in molecule.atoms:
        if start in seen:
            continue
        comp: list[oc.Atom] = []
        stack: list[oc.Atom] = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            comp.append(current)
            for nb in adj[current]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comp.sort(key=lambda a: index[a])
        components.append(comp)
    components.sort(key=lambda c: index[c[0]])
    return components


def _find_simple_cycles(
    adj: dict[oc.Atom, list[oc.Atom]], max_len: int = _MAX_RING_SIZE
) -> list[list[oc.Atom]]:
    """枚举 3..max_len 的简单环（以原子集合去重，顺序确定性）。"""
    atoms = list(adj)
    index: dict[oc.Atom, int] = {a: i for i, a in enumerate(atoms)}
    cycles: dict[frozenset[oc.Atom], list[oc.Atom]] = {}

    for start in atoms:
        path: list[oc.Atom] = [start]
        on_path: set[oc.Atom] = {start}

        def dfs(node: oc.Atom, depth: int) -> None:
            for nb in adj[node]:
                if nb is start:
                    if len(path) >= 3:
                        key = frozenset(path)
                        if key not in cycles:
                            cycles[key] = list(path)
                    continue
                if index[nb] < index[start] or nb in on_path:
                    continue
                if depth + 1 > max_len:
                    continue
                on_path.add(nb)
                path.append(nb)
                dfs(nb, depth + 1)
                path.pop()
                on_path.remove(nb)

        dfs(start, 1)
    return list(cycles.values())


def _are_bonded(atom1: oc.Atom, atom2: oc.Atom) -> bool:
    return any(bond.other(atom1) is atom2 for bond in atom1.bonds)


def _select_rings(
    cycles: list[list[oc.Atom]],
) -> list[list[oc.Atom]]:
    """贪心挑选互不冲突的环：短环优先；共享 >=3 个原子或两条以上共享边则拒绝。"""
    ordered = sorted(cycles, key=len)
    accepted: list[list[oc.Atom]] = []
    for ring in ordered:
        rset = set(ring)
        edge_shares: list[tuple[list[oc.Atom], tuple[oc.Atom, oc.Atom]]] = []
        conflict = False
        for acc in accepted:
            shared = rset & set(acc)
            if len(shared) >= 3:
                conflict = True
                break
            if len(shared) == 2:
                pair = tuple(sorted(shared, key=lambda a: id(a)))
                edge_shares.append((acc, pair))
        if conflict:
            continue
        if len(edge_shares) == 0:
            accepted.append(ring)
        elif len(edge_shares) == 1:
            acc, pair = edge_shares[0]
            if _are_bonded(pair[0], pair[1]):
                accepted.append(ring)
        # 多于一条共享边（复杂稠环）暂不支持，丢弃
    return accepted


def _group_ring_systems(
    rings: list[list[oc.Atom]], molecule: oc.Molecule
) -> list[list[list[oc.Atom]]]:
    """按共享边（>=2 个原子）把环聚成稠环系统；返回系统列表，顺序确定。"""
    atom_index: dict[oc.Atom, int] = {a: i for i, a in enumerate(molecule.atoms)}
    parent = list(range(len(rings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(set(rings[i]) & set(rings[j])) >= 2:
                union(i, j)

    groups: dict[int, list[list[oc.Atom]]] = {}
    for i, ring in enumerate(rings):
        groups.setdefault(find(i), []).append(ring)
    systems: list[list[list[oc.Atom]]] = []
    for group in groups.values():
        group.sort(key=lambda r: min(atom_index[a] for a in r))
        systems.append(group)
    systems.sort(key=lambda g: min(atom_index[a] for a in g[0]))
    return systems


def _polygon(
    ring: list[oc.Atom],
    center: tuple[float, float],
    start_angle: float,
) -> dict[oc.Atom, tuple[float, float]]:
    """把环按正多边形放置（顶点按环顺序依次排列）。"""
    n = len(ring)
    radius = BOND_LEN / (2.0 * math.sin(math.pi / n))
    cx, cy = center
    result: dict[oc.Atom, tuple[float, float]] = {}
    for i, atom in enumerate(ring):
        angle = start_angle + 2.0 * math.pi * i / n
        result[atom] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return result


def _centroid(
    positions: dict[oc.Atom, tuple[float, float]],
) -> tuple[float, float]:
    if not positions:
        return (0.0, 0.0)
    return (
        sum(x for x, _ in positions.values()) / len(positions),
        sum(y for _, y in positions.values()) / len(positions),
    )


def _place_fused_ring(
    ring: list[oc.Atom],
    atom_a: oc.Atom,
    atom_b: oc.Atom,
    pos_a: tuple[float, float],
    pos_b: tuple[float, float],
    other_centroid: tuple[float, float],
) -> dict[oc.Atom, tuple[float, float]]:
    """把与已放环共享边 (a,b) 的环放到共享边的另一侧（正多边形）。"""
    n = len(ring)
    radius = BOND_LEN / (2.0 * math.sin(math.pi / n))
    mx, my = (pos_a[0] + pos_b[0]) / 2.0, (pos_a[1] + pos_b[1]) / 2.0
    dx, dy = pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return _polygon(ring, (mx, my), -math.pi / 2.0)
    px, py = -dy / length, dx / length
    height = BOND_LEN / (2.0 * math.tan(math.pi / n))
    c1 = (mx + px * height, my + py * height)
    c2 = (mx - px * height, my - py * height)
    d1 = math.hypot(c1[0] - other_centroid[0], c1[1] - other_centroid[1])
    d2 = math.hypot(c2[0] - other_centroid[0], c2[1] - other_centroid[1])
    center = c1 if d1 >= d2 else c2

    ia, ib = ring.index(atom_a), ring.index(atom_b)
    step_dir = 1 if (ib - ia) % n == 1 else -1
    base_angle = math.atan2(pos_a[1] - center[1], pos_a[0] - center[0])
    result: dict[oc.Atom, tuple[float, float]] = {}
    for i in range(n):
        idx = (ia + step_dir * i) % n
        angle = base_angle + step_dir * 2.0 * math.pi * i / n
        result[ring[idx]] = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
    return result


def _place_ring_system(
    system: list[list[oc.Atom]],
    molecule: oc.Molecule,
    positions: dict[oc.Atom, tuple[float, float]],
    sub_offset: float,
) -> tuple[dict[oc.Atom, tuple[float, float]], float]:
    """放置一个稠环系统；返回新坐标与更新后的横向偏移。"""
    atom_index: dict[oc.Atom, int] = {a: i for i, a in enumerate(molecule.atoms)}
    local: dict[oc.Atom, tuple[float, float]] = {}

    # 已占用原子（与之前系统共享的螺环原子）
    shared = [a for ring in system for a in ring if a in positions]
    first_ring = min(system, key=lambda r: min(atom_index[a] for a in r))

    if shared:
        anchor = shared[0]
        pos_anchor = positions[anchor]
        global_centroid = _centroid(positions)
        direction = math.atan2(
            pos_anchor[1] - global_centroid[1],
            pos_anchor[0] - global_centroid[0],
        )
        n = len(first_ring)
        radius = BOND_LEN / (2.0 * math.sin(math.pi / n))
        center = (
            pos_anchor[0] + radius * math.cos(direction),
            pos_anchor[1] + radius * math.sin(direction),
        )
        base_angle = math.atan2(pos_anchor[1] - center[1], pos_anchor[0] - center[0])
        ia = first_ring.index(anchor)
        for i, atom in enumerate(first_ring):
            idx = (ia + i) % n
            angle = base_angle + 2.0 * math.pi * i / n
            local[first_ring[idx]] = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
    else:
        local.update(_polygon(first_ring, (sub_offset, 0.0), -math.pi / 2.0))

    placed: set[int] = set()
    first_id = id(first_ring)
    placed.add(first_id)
    queue: deque[list[oc.Atom]] = deque([first_ring])
    while queue:
        ring1 = queue.popleft()
        ring1_centroid = _centroid({a: local[a] for a in ring1 if a in local})
        for ring2 in system:
            if id(ring2) in placed:
                continue
            shared_atoms = set(ring1) & set(ring2)
            if len(shared_atoms) == 2:
                a, b = tuple(shared_atoms)
                if _are_bonded(a, b):
                    fused = _place_fused_ring(
                        ring2, a, b, local[a], local[b], ring1_centroid
                    )
                    local.update(fused)
                    placed.add(id(ring2))
                    queue.append(ring2)

    max_x = max(x for x, _ in local.values())
    return local, max_x + 2.0 * BOND_LEN


def _is_sp2(atom: oc.Atom) -> bool:
    """是否按 sp2（120°）布局：参与 π 体系，或存在键级 >= 2 的键。"""
    if any(atom in pi.atoms for pi in atom.belong.pi_systems):
        return True
    return any(bond.order >= 2 for bond in atom.bonds)


def _place_branches(
    molecule: oc.Molecule,
    adj: dict[oc.Atom, list[oc.Atom]],
    positions: dict[oc.Atom, tuple[float, float]],
    ring_centroid: dict[oc.Atom, tuple[float, float]],
    component: set[oc.Atom] | None = None,
) -> None:
    """BFS 放置所有非环原子：环原子向外、链原子顺延、起点向下。"""
    place_angle: dict[oc.Atom, float] = {}
    queue: deque[oc.Atom] = deque()
    for atom in molecule.atoms:
        if atom in positions and (component is None or atom in component):
            queue.append(atom)
    while queue:
        parent = queue.popleft()
        children = [nb for nb in adj[parent] if nb not in positions]
        if not children:
            continue
        if parent in ring_centroid:
            cx, cy = ring_centroid[parent]
            x, y = positions[parent]
            base = math.atan2(y - cy, x - cx)
        elif parent in place_angle:
            base = place_angle[parent]
        else:
            base = math.pi / 2.0  # 向下
        k = len(children)
        step = _SP2_ANGLE if _is_sp2(parent) else _SP3_ANGLE
        angles = [base + step * (i - (k - 1) / 2.0) for i in range(k)]
        px, py = positions[parent]
        for child, angle in zip(children, angles):
            positions[child] = (
                px + BOND_LEN * math.cos(angle),
                py + BOND_LEN * math.sin(angle),
            )
            place_angle[child] = angle
            queue.append(child)


def _refine(
    molecule: oc.Molecule,
    positions: dict[oc.Atom, tuple[float, float]],
    ring_atoms: set[oc.Atom],
    iterations: int = 100,
) -> None:
    """对非环原子做少量弹簧+斥力微调（环原子冻结，确定性）。"""
    free = [a for a in molecule.atoms if a not in ring_atoms]
    if not free:
        return
    atoms = molecule.atoms
    step = 0.08
    for _ in range(iterations):
        forces: dict[oc.Atom, tuple[float, float]] = {a: (0.0, 0.0) for a in free}
        for bond in molecule.bonds:
            a1, a2 = bond.atoms
            f1 = a1 in forces
            f2 = a2 in forces
            if not f1 and not f2:
                continue
            x1, y1 = positions[a1]
            x2, y2 = positions[a2]
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            force = (d - BOND_LEN) * 0.05
            ux, uy = dx / d, dy / d
            if f1:
                fx, fy = forces[a1]
                forces[a1] = (fx + ux * force, fy + uy * force)
            if f2:
                fx, fy = forces[a2]
                forces[a2] = (fx - ux * force, fy - uy * force)
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                a1, a2 = atoms[i], atoms[j]
                f1 = a1 in forces
                f2 = a2 in forces
                if not f1 and not f2:
                    continue
                x1, y1 = positions[a1]
                x2, y2 = positions[a2]
                dx, dy = x2 - x1, y2 - y1
                d = math.hypot(dx, dy)
                min_d = BOND_LEN * 1.05
                if d >= min_d or d < 1e-6:
                    continue
                push = (min_d - d) * 0.12
                ux, uy = dx / d, dy / d
                if f1:
                    fx, fy = forces[a1]
                    forces[a1] = (fx - ux * push, fy - uy * push)
                if f2:
                    fx, fy = forces[a2]
                    forces[a2] = (fx + ux * push, fy + uy * push)
        for atom in free:
            fx, fy = forces[atom]
            mx = max(-2.0, min(2.0, fx))
            my = max(-2.0, min(2.0, fy))
            x, y = positions[atom]
            positions[atom] = (x + mx * step, y + my * step)


def _normalize(
    positions: dict[oc.Atom, tuple[float, float]],
) -> dict[oc.Atom, tuple[float, float]]:
    if not positions:
        return {}
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return {a: (x - cx, y - cy) for a, (x, y) in positions.items()}


def compute_coordinates(
    molecule: oc.Molecule,
) -> dict[oc.Atom, tuple[float, float]]:
    """确定性 2D 布局：环 -> 稠合/螺环 -> 支链 BFS -> 非环原子微调 -> 居中。"""
    adj = _adjacency(molecule)
    components = _connected_components(molecule, adj)
    cycles = _find_simple_cycles(adj)
    rings = _select_rings(cycles)
    systems = _group_ring_systems(rings, molecule)

    positions: dict[oc.Atom, tuple[float, float]] = {}
    ring_centroid: dict[oc.Atom, tuple[float, float]] = {}
    offset_x = 0.0

    for component in components:
        comp_set = set(component)
        comp_systems = [
            sys for sys in systems if all(atom in comp_set for ring in sys for atom in ring)
        ]
        sub_offset = offset_x
        placed_any = False
        for system in comp_systems:
            local, sub_offset = _place_ring_system(
                system, molecule, positions, sub_offset
            )
            centroid = _centroid(local)
            for atom in local:
                ring_centroid[atom] = centroid
            positions.update(local)
            placed_any = True
        if not placed_any:
            positions[component[0]] = (sub_offset, 0.0)
        _place_branches(
            molecule, adj, positions, ring_centroid, component=comp_set
        )
        comp_positions = [positions[a] for a in component]
        offset_x = max(x for x, _ in comp_positions) + 2.0 * BOND_LEN

    _place_branches(molecule, adj, positions, ring_centroid)
    _refine(molecule, positions, set(ring_centroid))
    return _normalize(positions)


# ------- 载荷 -------


def _subscript(count: int) -> str:
    return "".join(_SUBSCRIPTS[int(d)] for d in str(count))


def _atom_label(atom: oc.Atom) -> str:
    """原子标签：碳显示 CHn；杂原子显示符号 + 隐氢（显式 H 节点单独绘制）。"""
    if atom.name == "c":
        if atom.implicit_h == 0:
            return "C"
        return "C" + "H" + _subscript(atom.implicit_h)
    if atom.name == "h":
        return "H"
    label = atom.name.capitalize()
    if atom.implicit_h:
        label += "H" + _subscript(atom.implicit_h)
    return label


def _pi_kind(pi: oc.PiSystem) -> str | None:
    counts: dict[str, int] = {}
    for atom in pi.atoms:
        counts[atom.name] = counts.get(atom.name, 0) + 1
    if len(pi.atoms) == 6 and counts.get("c") == 6:
        return "benzene"
    if len(pi.atoms) == 3 and counts.get("n") == 1 and counts.get("o") == 2:
        return "nitro"
    return None


def _formula_string(formula: dict[str, int]) -> str:
    parts: list[str] = []
    for element, count in formula.items():
        if count <= 0:
            continue
        parts.append(element.capitalize() + (str(count) if count > 1 else ""))
    return "".join(parts)


def molecule_to_payload(molecule: oc.Molecule, source: str = "") -> dict[str, Any]:
    """生成前端渲染所需的完整 JSON 载荷。"""
    coords = compute_coordinates(molecule)
    atom_index: dict[oc.Atom, int] = {a: i for i, a in enumerate(molecule.atoms)}
    ring_atoms = _find_placed_ring_atoms(molecule)

    display_orders: dict[tuple[oc.Atom, oc.Atom], int] = {}
    for pi in molecule.pi_systems:
        kind = _pi_kind(pi)
        if kind == "nitro":
            n_atom = next(a for a in pi.atoms if a.name == "n")
            o_atom = next(a for a in pi.atoms if a.name == "o")
            for bond in molecule.bonds:
                if set(bond.atoms) == {n_atom, o_atom}:
                    display_orders[tuple(sorted(bond.atoms, key=lambda a: id(a)))] = 2
                    break

    atoms: list[dict[str, Any]] = []
    for i, atom in enumerate(molecule.atoms):
        x, y = coords[atom]
        atoms.append(
            {
                "id": i,
                "element": atom.name,
                "x": round(x, 3),
                "y": round(y, 3),
                "label": _atom_label(atom),
                "h": atom.implicit_h,
                "active": isinstance(atom, oc.ActiveH),
                "ring": atom in ring_atoms,
            }
        )

    bonds: list[dict[str, Any]] = []
    for bond in molecule.bonds:
        a1, a2 = bond.atoms
        key = tuple(sorted(bond.atoms, key=lambda a: id(a)))
        bonds.append(
            {
                "a": atom_index[a1],
                "b": atom_index[a2],
                "order": bond.order,
                "display_order": display_orders.get(key, bond.order),
            }
        )

    pi_systems: list[dict[str, Any]] = []
    for pi in molecule.pi_systems:
        pi_systems.append(
            {
                "atoms": [atom_index[a] for a in pi.atoms],
                "dbe": pi.dbe,
                "display": _pi_kind(pi),
            }
        )

    groups: list[dict[str, Any]] = []
    for group in oc_features.functional_groups(molecule):
        groups.append(
            {
                "name": group.name,
                "category": group.category,
                "atoms": [atom_index[a] for a in group.atoms],
            }
        )

    return {
        "name": molecule.name,
        "source": source,
        "formula": _formula_string(molecule.formula),
        "unsaturation": molecule.unsaturation,
        "ring_count": molecule.ring_count,
        "component_count": molecule.component_count,
        "atom_count": len(molecule.atoms),
        "bond_count": len(molecule.bonds),
        "pi_count": len(molecule.pi_systems),
        "groups": groups,
        "atoms": atoms,
        "bonds": bonds,
        "pi_systems": pi_systems,
    }


def _find_placed_ring_atoms(molecule: oc.Molecule) -> set[oc.Atom]:
    """找出参与环布局的原子（用于前端隐藏环上碳的标签）。"""
    adj = _adjacency(molecule)
    rings = _select_rings(_find_simple_cycles(adj))
    return {a for ring in rings for a in ring}
