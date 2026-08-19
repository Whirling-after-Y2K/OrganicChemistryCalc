"""同分异构体分析（结构枚举，无第三方依赖）。

设计（基团方案）：
- 苯环：分子式可含苯环时（C >= 6 且不饱和度 >= 4），把苯环作为预建的
  固定基团（6 元碳环单键 + PiSystem，6 个单键槽位）加入枚举，普通原子
  递归里不再出现凯库勒环归一化路径；同一苯环的两个槽位之间禁止直接成键。
- 硝基：作为固定基团（N-O 单键 + PiSystem，N 槽位容量 1，只能接受一根
  外部单键且另一端必须为碳），枚举前按分子式预检（N/O 预算、不饱和度、
  是否存在可挂载的碳）。
- 普通模式（无基团）用于不可能含苯环的分子式（C < 6 或不饱和度 < 4），
  按图论全集枚举；可含苯环的分子式只枚举"含苯环/硝基"的结构族，
  符合高中化学口径（不含 Dewar 苯等罕见价键异构）。
- 稠环芳烃（萘式）若被枚举，基团模式下以"单环 π + 其余显式双键"形式
  列出，v1 不保证规范表示。
- 剪枝：氢数预算（当前隐氢数不能低于目标）、不饱和度预算（基团贡献 +
  多重键 + 已形成环数不能超过目标）、孤立原子剪枝。
- 输出：全部同分异构体（含输入结构本身的隐氢形式），按结构指纹稳定排序。
"""

from __future__ import annotations

import os
from pathlib import Path

import oc_io
import organic_chemistry as oc

MAX_HEAVY_ATOMS: int = 12

# 单价元素：隐氢模型无法承载氢（如 HF/HCl），此类分子退化为仅输入本身
_MONOVALENT: frozenset[str] = frozenset({"f", "cl", "br", "i"})


def _display_formula(formula: dict[str, int]) -> str:
    """分子式文本，如 {'c':4,'h':10,'o':1} -> 'C4H10O'。"""
    parts: list[str] = []
    for element, count in formula.items():
        if count <= 0:
            continue
        if count == 1:
            parts.append(element.capitalize())
        else:
            parts.append(f"{element.capitalize()}{count}")
    return "".join(parts)


def _formula_key(formula: dict[str, int]) -> str:
    """目录友好的小写分子式键，如 'c4h10o'。"""
    parts: list[str] = []
    for element, count in formula.items():
        if count <= 0:
            continue
        parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


def _heavy_counts(molecule: oc.Molecule) -> dict[str, int]:
    """重原子多重集：忽略显式氢节点（H 一律按隐氢处理）。"""
    counts: dict[str, int] = {}
    for atom in molecule.atoms:
        if atom.name == 'h':
            continue
        counts[atom.name] = counts.get(atom.name, 0) + 1
    return counts


def _formula_dbe(formula: dict[str, int]) -> int:
    """按分子式计算不饱和度（用于苯环/硝基预检）。"""
    carbon = formula.get('c', 0)
    hydrogen = formula.get('h', 0)
    nitrogen = formula.get('n', 0)
    halogens = (
        formula.get('f', 0)
        + formula.get('cl', 0)
        + formula.get('br', 0)
        + formula.get('i', 0)
    )
    return (2 * carbon + 2 + nitrogen - hydrogen - halogens) // 2


def find_isomers(molecule: oc.Molecule) -> list[oc.Molecule]:
    """返回分子的全部同分异构体（含输入结构本身的隐氢形式）。

    - 结果按 (tuple(feature), 生成序) 稳定排序，已按结构判等去重；
    - 重原子数超过 MAX_HEAVY_ATOMS 抛 ValueError；
    - 无法用隐氢模型表示的分子（如 H2、HF）退化为仅返回输入本身；
    - 可含苯环的分子式只枚举含苯环/硝基的结构族（高中口径）。
    """
    molecule.validate()
    target = dict(molecule.formula)
    heavy = _heavy_counts(molecule)
    total_heavy = sum(heavy.values())
    if total_heavy > MAX_HEAVY_ATOMS:
        raise ValueError(f"重原子数 {total_heavy} 超过上限 {MAX_HEAVY_ATOMS}")
    if total_heavy == 0 or (
        all(element in _MONOVALENT for element in heavy) and target.get('h', 0) > 0
    ):
        return [oc.copy_molecule(molecule)]

    dbe = _formula_dbe(target)
    collected: list[oc.Molecule] = []
    if heavy.get('c', 0) >= 6 and dbe >= 4:
        for k_b in range(1, heavy.get('c', 0) // 6 + 1):
            budget = dbe - 4 * k_b
            if budget < 0:
                continue
            max_nitro = min(heavy.get('n', 0), heavy.get('o', 0) // 2, budget)
            for k_n in range(max_nitro + 1):
                _collect_mode(k_b, k_n, heavy, target, collected)
        if not collected:
            # 罕见：分子式满足可含苯环但不存在含苯环结构 → 回退普通模式
            max_nitro = min(heavy.get('n', 0), heavy.get('o', 0) // 2, dbe)
            for k_n in range(max_nitro + 1):
                if k_n > 0 and heavy.get('c', 0) == 0:
                    continue
                _collect_mode(0, k_n, heavy, target, collected)
    else:
        max_nitro = min(heavy.get('n', 0), heavy.get('o', 0) // 2, dbe)
        for k_n in range(max_nitro + 1):
            if k_n > 0 and heavy.get('c', 0) == 0:
                continue
            _collect_mode(0, k_n, heavy, target, collected)

    buckets: dict[tuple[str, ...], list[oc.Molecule]] = {}
    results: list[oc.Molecule] = []
    for candidate in collected:
        key = tuple(candidate.feature)
        bucket = buckets.setdefault(key, [])
        if any(candidate == other for other in bucket):
            continue
        bucket.append(candidate)
        results.append(candidate)
    results.sort(key=lambda item: tuple(item.feature))
    return results


def _collect_mode(
    k_b: int,
    k_n: int,
    heavy: dict[str, int],
    target: dict[str, int],
    collected: list[oc.Molecule],
) -> None:
    """枚举含 k_b 个苯环基团、k_n 个硝基基团的所有候选。

    递归原子清单：先苯环槽位（6k_b 个，容量 1），再硝基 N（k_n 个，容量 1），
    最后按元素排序的剩余自由原子。苯环/硝基内部结构固定，不参与成键递归。
    """
    elements: list[str] = []
    limits: list[int] = []
    base: list[int] = []
    h0: list[int] = []
    is_nitro: list[bool] = []
    is_slot: list[bool] = []
    ring_id: list[int] = []
    for index in range(6 * k_b):
        elements.append('c')
        limits.append(4)
        base.append(3)          # 2 根环单键 + 1 个 π 槽位
        h0.append(1)
        is_nitro.append(False)
        is_slot.append(True)
        ring_id.append(index // 6)
    for _ in range(k_n):
        elements.append('n')
        limits.append(4)        # 硝基型 [N,O,O] 中的 N 允许 4 个槽位
        base.append(3)          # 2 根 N-O 单键 + 1 个 π 槽位
        h0.append(1)
        is_nitro.append(True)
        is_slot.append(False)
        ring_id.append(-1)
    remaining: dict[str, int] = {element: count for element, count in heavy.items()}
    remaining['c'] = remaining.get('c', 0) - 6 * k_b
    remaining['n'] = remaining.get('n', 0) - k_n
    remaining['o'] = remaining.get('o', 0) - 2 * k_n
    for element in sorted(remaining):
        for _ in range(remaining[element]):
            elements.append(element)
            limit = oc.CHEMISTRY_BOND_DICT[element]  # type: ignore
            limits.append(limit)
            base.append(0)
            h0.append(limit if limit >= 2 else 0)
            is_nitro.append(False)
            is_slot.append(False)
            ring_id.append(-1)
    n = len(elements)
    if n == 0:
        return
    target_h = target.get('h', 0)
    target_dbe = _formula_dbe(target)
    base_dbe = 4 * k_b + k_n

    used: list[int] = list(base)
    bond_count: list[int] = [0] * n
    bonds: list[tuple[int, int, int]] = []
    partner: list[int] = [-1] * n

    def current_h() -> int:
        total = 0
        for index in range(n):
            total += max(0, h0[index] - (used[index] - base[index]))
        return total

    def count_rings() -> int:
        """当前已形成环数（含基团初始连通带来的环，通过并查集判定）。"""
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for index in range(n):
            if not is_slot[index]:
                continue
            for other in range(index + 1, n):
                if is_slot[other] and ring_id[index] == ring_id[other]:
                    parent[find(other)] = find(index)
        rings = 0
        for a, b, _order in bonds:
            root_a, root_b = find(a), find(b)
            if root_a == root_b:
                rings += 1
            else:
                parent[root_b] = root_a
        return rings

    def build_leaf() -> None:
        molecule = oc.Molecule()
        ring_atoms: list[oc.Atom] = []
        for _ in range(k_b):
            ring = [oc.Atom('c', molecule) for _ in range(6)]
            for i in range(6):
                oc.add_bond(ring[i], ring[(i + 1) % 6])
            oc.add_pi_system(ring)
            ring_atoms.extend(ring)
        nitro_groups: list[tuple[oc.Atom, oc.Atom, oc.Atom]] = []
        for _ in range(k_n):
            n_atom = oc.Atom('n', molecule)
            o1 = oc.Atom('o', molecule)
            o2 = oc.Atom('o', molecule)
            oc.add_bond(n_atom, o1)
            oc.add_bond(n_atom, o2)
            oc.add_pi_system([n_atom, o1, o2])
            nitro_groups.append((n_atom, o1, o2))
        free_atoms: list[oc.Atom] = []
        for element in sorted(remaining):
            for _ in range(remaining[element]):
                free_atoms.append(oc.Atom(element, molecule))
        atoms = ring_atoms + [group[0] for group in nitro_groups] + free_atoms
        try:
            for a, b, order in bonds:
                oc.add_bond(atoms[a], atoms[b], order)
            # 硝基规则：每个硝基 N 恰好一根外部单键，且另一端必须为碳
            for n_atom, o1, o2 in nitro_groups:
                exo = [bond for bond in n_atom.bonds if bond.other(n_atom) not in (o1, o2)]
                if len(exo) != 1 or exo[0].other(n_atom).name != 'c':
                    return
            if molecule.component_count != 1:
                return
            molecule.validate()
            if molecule.formula != target:
                return
        except ValueError:
            return
        collected.append(molecule)

    def backtrack(i: int, j: int) -> None:
        if i == n:
            build_leaf()
            return
        h_now = current_h()
        if h_now < target_h:
            return
        if h_now == target_h:
            # 再成键只会减少氢数：剩余原子对只能全为 0，直接按当前状态收尾
            build_leaf()
            return
        if j == n:
            # 原子 i 的全部对外键已确定：剪枝孤立自由原子 / 硝基 N 缺挂载
            if is_nitro[i]:
                if partner[i] < 0 or elements[partner[i]] != 'c':
                    return
            elif not is_slot[i] and n > 1 and bond_count[i] == 0:
                return
            backtrack(i + 1, i + 2)
            return
        if base_dbe + sum(order - 1 for _, _, order in bonds) + count_rings() > target_dbe:
            return
        if is_slot[i] and is_slot[j] and ring_id[i] == ring_id[j]:
            # 同一苯环的两个槽位之间禁止成键（保持苯环为干净的单环）
            backtrack(i, j + 1)
            return
        max_order = min(3, limits[i] - used[i], limits[j] - used[j])
        for order in range(max_order + 1):
            if order > 0:
                used[i] += order
                used[j] += order
                bond_count[i] += 1
                bond_count[j] += 1
                bonds.append((i, j, order))
                if is_nitro[i]:
                    partner[i] = j
            backtrack(i, j + 1)
            if order > 0:
                bonds.pop()
                bond_count[i] -= 1
                bond_count[j] -= 1
                used[i] -= order
                used[j] -= order
                if is_nitro[i]:
                    partner[i] = -1

    backtrack(0, 1)


def isomer_report(molecule: oc.Molecule) -> str:
    """控制台文字报告：分子式、总数、每个异构体的编号/分子式/不饱和度/环数。"""
    isomers = find_isomers(molecule)
    lines: list[str] = [
        f"分子式：{_display_formula(molecule.formula)}",
        f"同分异构体总数：{len(isomers)}",
        "",
    ]
    for index, candidate in enumerate(isomers, 1):
        lines.append(
            f"异构体 {index:02d}：{_display_formula(candidate.formula)}，"
            f"不饱和度 {candidate.unsaturation}，环数 {candidate.ring_count}"
        )
    return "\n".join(lines)


def save_isomers(molecule: oc.Molecule, directory: str | os.PathLike[str]) -> list[Path]:
    """把全部异构体保存为构建脚本（oc_io），返回文件路径列表。

    目录结构：<directory>/<公式键>/isomer_01.py、isomer_02.py...
    """
    isomers = find_isomers(molecule)
    folder = Path(directory) / _formula_key(molecule.formula)
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, candidate in enumerate(isomers, 1):
        path = folder / f"isomer_{index:02d}.py"
        oc_io.save_molecule(candidate, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    demo = oc.Molecule()
    chain = [oc.Atom('c', demo) for _ in range(4)]
    oc.connect(chain)
    oc.add_bond(chain[0], oc.Atom('o', demo))
    print(isomer_report(demo))
