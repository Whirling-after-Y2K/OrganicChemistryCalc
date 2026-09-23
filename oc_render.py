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
_SYSTEM_GAP: float = 2.0 * BOND_LEN      # 独立安放的环系统之间的水平间距
_MIN_GAP: float = 1.05 * BOND_LEN        # 非键原子之间允许的最小间距
_ROTATION_STEP: float = math.pi / 6.0    # 环系统安放时的候选方向步长（30°）
_ROTATION_SPAN: int = 6                  # 候选方向向两侧展开的步数（合计 ±180°）


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

    # 顶点编号要同时对齐两个方向：角度方向由共享边 a->b 相对环心的转角决定，
    # 环表方向则取决于 b 在环表里是紧跟 a 还是前接 a。只有两者取齐，
    # 环上相邻的原子才会落到相邻顶点上，共有原子也才正好落回原位。
    start = ring.index(atom_a)
    base_angle = math.atan2(pos_a[1] - center[1], pos_a[0] - center[0])
    if ring[(start + 1) % n] is atom_b:
        list_dir = 1
    elif ring[(start - 1) % n] is atom_b:
        list_dir = -1
    else:  # 共享边两端在环表里不相邻（不应发生），退回整体正多边形
        return _polygon(ring, center, base_angle)
    other_angle = math.atan2(pos_b[1] - center[1], pos_b[0] - center[0])
    turn = (other_angle - base_angle + math.pi) % (2.0 * math.pi) - math.pi
    step_dir = 1 if turn >= 0.0 else -1
    result: dict[oc.Atom, tuple[float, float]] = {}
    for i in range(n):
        angle = base_angle + step_dir * 2.0 * math.pi * i / n
        atom = ring[(start + list_dir * i) % n]
        result[atom] = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
    return result


def _system_members(system: list[list[oc.Atom]]) -> set[oc.Atom]:
    """环系统覆盖的全部原子。"""
    return {atom for ring in system for atom in ring}


def _first_ring(
    system: list[list[oc.Atom]], molecule: oc.Molecule
) -> list[oc.Atom]:
    """系统内原子编号最小的环，作为生长起点（保证确定性）。"""
    atom_index: dict[oc.Atom, int] = {a: i for i, a in enumerate(molecule.atoms)}
    return min(system, key=lambda ring: min(atom_index[a] for a in ring))


def _shared_edge(
    shared: set[oc.Atom], index: dict[oc.Atom, int]
) -> tuple[oc.Atom, oc.Atom] | tuple[None, None]:
    """把两个环的共有原子按分子编号排成共享边；不相邻时返回 (None, None)。"""
    if len(shared) != 2:
        return (None, None)
    atom_a, atom_b = sorted(shared, key=lambda a: index[a])
    if not _are_bonded(atom_a, atom_b):
        return (None, None)
    return (atom_a, atom_b)


def _grow_ring_system(
    system: list[list[oc.Atom]],
    seed_ring: list[oc.Atom],
    local: dict[oc.Atom, tuple[float, float]],
    index: dict[oc.Atom, int],
) -> dict[oc.Atom, tuple[float, float]]:
    """从 seed_ring 出发，按共享边把同一系统的其余环逐个接上。"""
    placed: set[int] = {id(seed_ring)}
    queue: deque[list[oc.Atom]] = deque([seed_ring])
    while queue:
        ring1 = queue.popleft()
        ring1_centroid = _centroid({a: local[a] for a in ring1 if a in local})
        for ring2 in system:
            if id(ring2) in placed:
                continue
            shared_atoms = set(ring1) & set(ring2)
            if len(shared_atoms) != 2:
                continue
            atom_a, atom_b = _shared_edge(shared_atoms, index)
            if atom_a is None:
                continue  # 只共享两个不相邻原子的桥环暂不支持
            local.update(
                _place_fused_ring(
                    ring2, atom_a, atom_b, local[atom_a], local[atom_b], ring1_centroid
                )
            )
            placed.add(id(ring2))
            queue.append(ring2)
    return local


def _attachment(
    system: list[list[oc.Atom]],
    molecule: oc.Molecule,
    adj: dict[oc.Atom, list[oc.Atom]],
    positions: dict[oc.Atom, tuple[float, float]],
) -> tuple[oc.Atom, oc.Atom] | None:
    """找出环系统与已放置结构的连接：返回 (外部原子, 系统内原子)。

    螺环的共用原子、以及环与环之间直接成键的情况都在这里识别。
    """
    members = _system_members(system)
    for atom in molecule.atoms:
        if atom not in members:
            continue
        for neighbor in adj[atom]:
            if neighbor in positions and neighbor not in members:
                return neighbor, atom
    return None


def _outward_direction(
    atom: oc.Atom,
    adj: dict[oc.Atom, list[oc.Atom]],
    positions: dict[oc.Atom, tuple[float, float]],
) -> tuple[float, float]:
    """从 atom 的已放置邻居指向外侧的单位方向（新结构的外延方向）。"""
    x, y = positions[atom]
    sx = 0.0
    sy = 0.0
    for neighbor in adj[atom]:
        if neighbor not in positions:
            continue
        nx, ny = positions[neighbor]
        sx += x - nx
        sy += y - ny
    norm = math.hypot(sx, sy)
    if norm < 1e-9:
        return (1.0, 0.0)
    return (sx / norm, sy / norm)


def _collision_score(
    local: dict[oc.Atom, tuple[float, float]],
    positions: dict[oc.Atom, tuple[float, float]],
) -> float:
    """候选布局与已放置原子的冲突程度（0 表示互不重叠）。"""
    score = 0.0
    for atom, (x, y) in local.items():
        for other, (ox, oy) in positions.items():
            if other in local or _are_bonded(atom, other):
                continue
            distance = math.hypot(x - ox, y - oy)
            if distance < _MIN_GAP:
                score += (_MIN_GAP - distance) ** 2
    return score


def _heading_candidates() -> list[float]:
    """候选方向序列：先保持外延方向，再按 ±30° 逐步试探。"""
    steps = [0]
    for step in range(1, _ROTATION_SPAN + 1):
        steps.extend([step, -step])
    return [_ROTATION_STEP * step for step in steps]


def _layout_system(
    system: list[list[oc.Atom]],
    molecule: oc.Molecule,
    attach_atom: oc.Atom,
    target: tuple[float, float],
    heading: float,
) -> dict[oc.Atom, tuple[float, float]]:
    """把系统内的 attach_atom 放到 target，系统整体朝 heading 方向展开。"""
    seed_ring = next((ring for ring in system if attach_atom in ring), None)
    if seed_ring is None:
        seed_ring = _first_ring(system, molecule)
    ordered = list(seed_ring)
    if attach_atom in ordered:
        start = ordered.index(attach_atom)
        ordered = ordered[start:] + ordered[:start]
    n = len(ordered)
    radius = BOND_LEN / (2.0 * math.sin(math.pi / n))
    center = (
        target[0] - radius * math.cos(heading),
        target[1] - radius * math.sin(heading),
    )
    local: dict[oc.Atom, tuple[float, float]] = _polygon(ordered, center, heading)
    index: dict[oc.Atom, int] = {a: i for i, a in enumerate(molecule.atoms)}
    return _grow_ring_system(system, seed_ring, local, index)


def _attach_system(
    system: list[list[oc.Atom]],
    molecule: oc.Molecule,
    adj: dict[oc.Atom, list[oc.Atom]],
    positions: dict[oc.Atom, tuple[float, float]],
    anchor_atom: oc.Atom,
    attach_atom: oc.Atom,
) -> dict[oc.Atom, tuple[float, float]]:
    """把环系统贴着已放置原子安放：枚举候选方向，取冲突最小的布局。"""
    reference = attach_atom if attach_atom in positions else anchor_atom
    base = _outward_direction(reference, adj, positions)
    base_angle = math.atan2(base[1], base[0])
    if attach_atom in positions:
        target = positions[attach_atom]  # 螺环：共用原子保持原位
    else:
        ax, ay = positions[anchor_atom]
        target = (ax + BOND_LEN * base[0], ay + BOND_LEN * base[1])

    best: dict[oc.Atom, tuple[float, float]] = {}
    best_score = math.inf
    for turn in _heading_candidates():
        local = _layout_system(system, molecule, attach_atom, target, base_angle + turn)
        score = _collision_score(local, positions)
        if score < best_score - 1e-9:
            best, best_score = local, score
        if score <= 0.0:
            break
    return best


def _place_system_freestanding(
    system: list[list[oc.Atom]],
    molecule: oc.Molecule,
    sub_offset: float,
) -> tuple[dict[oc.Atom, tuple[float, float]], float]:
    """独立安放一个环系统：整体平移到 sub_offset 右侧，返回坐标与新偏移。"""
    first_ring = _first_ring(system, molecule)
    seed = _polygon(first_ring, (0.0, 0.0), -math.pi / 2.0)
    shift = sub_offset - min(x for x, _ in seed.values())
    seed = {atom: (x + shift, y) for atom, (x, y) in seed.items()}
    index: dict[oc.Atom, int] = {a: i for i, a in enumerate(molecule.atoms)}
    local = _grow_ring_system(system, first_ring, seed, index)
    max_x = max(x for x, _ in local.values())
    return local, max(sub_offset, max_x + _SYSTEM_GAP)


def _is_sp2(atom: oc.Atom) -> bool:
    """是否按 sp2（120°）布局：参与 π 体系，或存在键级 >= 2 的键。"""
    if any(atom in pi.atoms for pi in atom.belong.pi_systems):
        return True
    return any(bond.order >= 2 for bond in atom.bonds)


def _place_children(
    parent: oc.Atom,
    adj: dict[oc.Atom, list[oc.Atom]],
    positions: dict[oc.Atom, tuple[float, float]],
    place_angle: dict[oc.Atom, float],
    ring_centroid: dict[oc.Atom, tuple[float, float]],
) -> list[oc.Atom]:
    """把 parent 尚未定位的邻居按理想键角扇形展开，返回新放置的原子。

    环原子沿半径向外、链原子顺延父键方向、链条起点朝下；
    多个邻居关于基准角对称排开，避免全挤在一边。
    """
    children = [nb for nb in adj[parent] if nb not in positions]
    if not children:
        return []
    if parent in ring_centroid:
        cx, cy = ring_centroid[parent]
        x, y = positions[parent]
        base = math.atan2(y - cy, x - cx)
    elif parent in place_angle:
        base = place_angle[parent]
    else:
        base = math.pi / 2.0  # 链条起点向下
    step = _SP2_ANGLE if _is_sp2(parent) else _SP3_ANGLE
    px, py = positions[parent]
    for i, child in enumerate(children):
        angle = base + step * (i - (len(children) - 1) / 2.0)
        positions[child] = (
            px + BOND_LEN * math.cos(angle),
            py + BOND_LEN * math.sin(angle),
        )
        place_angle[child] = angle
    return children


def _refine(
    molecule: oc.Molecule,
    positions: dict[oc.Atom, tuple[float, float]],
    ring_atoms: set[oc.Atom],
    iterations: int = 300,
) -> None:
    """对非环原子做键长 + 间距投影松弛（环原子冻结，确定性）。

    每一轮先把偏离的键长补回 BOND_LEN，再把间距不足的非键原子推开；
    位移按可移动端均摊，环原子不动。没有明显冲突即提前收工，
    因此本来就排得开的分子布局保持原样。
    """
    free = [a for a in molecule.atoms if a not in ring_atoms]
    if not free:
        return
    atoms = molecule.atoms
    for _ in range(iterations):
        worst_gap = 0.0
        for bond in molecule.bonds:
            atom1, atom2 = bond.atoms
            movable = [a for a in (atom1, atom2) if a not in ring_atoms]
            if not movable:
                continue
            x1, y1 = positions[atom1]
            x2, y2 = positions[atom2]
            dx, dy = x2 - x1, y2 - y1
            distance = math.hypot(dx, dy)
            if distance < 1e-6:
                continue
            correction = (distance - BOND_LEN) / (2.0 * len(movable))
            ux, uy = dx / distance, dy / distance
            for atom in movable:
                sign = 1.0 if atom is atom1 else -1.0
                x, y = positions[atom]
                positions[atom] = (
                    x + sign * ux * correction,
                    y + sign * uy * correction,
                )
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                atom1, atom2 = atoms[i], atoms[j]
                if _are_bonded(atom1, atom2):
                    continue
                movable = [a for a in (atom1, atom2) if a not in ring_atoms]
                if not movable:
                    continue
                x1, y1 = positions[atom1]
                x2, y2 = positions[atom2]
                dx, dy = x2 - x1, y2 - y1
                distance = math.hypot(dx, dy)
                if distance >= _MIN_GAP:
                    continue
                worst_gap = max(worst_gap, _MIN_GAP - distance)
                if distance < 1e-6:
                    ux, uy = 1.0, 0.0  # 完全重合时给一个确定方向
                else:
                    ux, uy = dx / distance, dy / distance
                push = (_MIN_GAP - distance) / len(movable)
                for atom in movable:
                    sign = 1.0 if atom is atom1 else -1.0
                    x, y = positions[atom]
                    positions[atom] = (x + sign * ux * push, y + sign * uy * push)
        if worst_gap <= 1e-3:
            break


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
    """确定性 2D 布局：以种子环系统为起点按连通顺序生长。

    每个连通分量先安放编号最小的环系统（无环分量则安放编号最小的原子），
    随后广度优先向外生长：遇到还没安放的环系统就整个接上——稠环、螺环、
    环与环直接成键、以及环经链与环相连都在这一步处理；其余邻居按理想
    键角扇形展开。最后做间距松弛并居中。
    """
    adj = _adjacency(molecule)
    index: dict[oc.Atom, int] = {a: i for i, a in enumerate(molecule.atoms)}
    components = _connected_components(molecule, adj)
    cycles = _find_simple_cycles(adj)
    rings = _select_rings(cycles)
    systems = _group_ring_systems(rings, molecule)
    system_of: dict[oc.Atom, int] = {
        atom: no
        for no, system in enumerate(systems)
        for atom in _system_members(system)
    }

    positions: dict[oc.Atom, tuple[float, float]] = {}
    ring_centroid: dict[oc.Atom, tuple[float, float]] = {}
    place_angle: dict[oc.Atom, float] = {}
    placed: set[int] = set()
    queue: deque[oc.Atom] = deque()

    def install(
        system_no: int, local: dict[oc.Atom, tuple[float, float]]
    ) -> None:
        """登记一个环系统：记下环心、并入全局坐标并推入队列。"""
        centroid = _centroid(local)
        for atom in local:
            ring_centroid[atom] = centroid
        positions.update(local)
        for atom in local:
            queue.append(atom)
        placed.add(system_no)

    offset_x = 0.0
    for component in components:
        seeded = [a for a in component if a in system_of]
        if seeded:
            seed_atom = min(seeded, key=lambda a: index[a])
            seed_no = system_of[seed_atom]
            local, _ = _place_system_freestanding(
                systems[seed_no], molecule, offset_x
            )
            install(seed_no, local)
        else:
            positions[component[0]] = (offset_x, 0.0)
            queue.append(component[0])

        while queue:
            parent = queue.popleft()
            # 环系统优先：邻居属于尚未安放系统时，先把整个系统接上
            fresh: list[int] = []
            for child in (nb for nb in adj[parent] if nb not in positions):
                no = system_of.get(child)
                if no is not None and no not in placed and no not in fresh:
                    fresh.append(no)
            for no in sorted(fresh):
                link = _attachment(systems[no], molecule, adj, positions)
                if link is None:
                    continue  # 邻居本身已带一个已放置原子，不会到这里
                anchor_atom, attach_atom = link
                install(
                    no,
                    _attach_system(
                        systems[no], molecule, adj, positions,
                        anchor_atom, attach_atom,
                    ),
                )
            for child in _place_children(
                parent, adj, positions, place_angle, ring_centroid
            ):
                queue.append(child)

        offset_x = (
            max(x for x, _ in (positions[a] for a in component)) + _SYSTEM_GAP
        )

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
        suffix = _subscript(atom.implicit_h) if atom.implicit_h > 1 else ""
        return "C" + "H" + suffix
    if atom.name == "h":
        return "H"
    label = atom.name.capitalize()
    if atom.implicit_h:
        label += "H" + (_subscript(atom.implicit_h) if atom.implicit_h > 1 else "")
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
        elif kind == "benzene":
            # 凯库勒式显示：按环顺序每隔一条边显示为双键（纯显示，不改模型）
            pi_set = frozenset(pi.atoms)
            for cycle in _find_simple_cycles(_adjacency(molecule)):
                if len(cycle) == 6 and set(cycle) == pi_set:
                    for index in (0, 2, 4):
                        atom1, atom2 = cycle[index], cycle[(index + 1) % 6]
                        display_orders[
                            tuple(sorted((atom1, atom2), key=lambda a: id(a)))
                        ] = 2
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
        "equivalent_hydrogen_groups": molecule.equivalent_hydrogen_groups,
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
