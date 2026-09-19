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
- 高中口径默认禁止苯环/片段外部额外成环；allow_extra_rings=True 可恢复
  允许额外环的完整图论行为（仍受搜索规模守卫限制）。
- 稠环芳烃（萘式）若被枚举，基团模式下以"单环 π + 其余显式双键"形式
  列出，v1 不保证规范表示。
- 剪枝：氢数预算（当前隐氢数不能低于目标）、不饱和度预算（基团贡献 +
  多重键 + 已形成环数不能超过目标）、孤立原子剪枝。
- 基团约束：可要求搜索结果必须含有指定基团（复用 oc_features 的基团检测）。
  苯环/硝基在模式层剪枝，其余基团在 DFS 中用必要条件剪枝（宁可少剪，
  不可剪错），枚举后统一用 functional_groups 过滤作为最终准确保证。
- 性能：可片段化的多原子基团（羧基/酯基/醛基/酮羰基/酰胺键/碳碳双键/三键）
  作为预建片段参与枚举，约束搜索秒级完成；搜索规模守卫（时间/节点双上限）
  防止无约束大分子枚举挂死。
- 等位氢约束：可要求异构体的等位氢模式（NMR 峰面积比）与期望一致，
  一律按公约数约分后比较；构建前用逐原子氢数多重集做必要条件预检。
- 输出：全部同分异构体（含输入结构本身的隐氢形式），按结构指纹稳定排序。
"""

from __future__ import annotations

import os
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import oc_io
import oc_features
import organic_chemistry as oc

MAX_HEAVY_ATOMS: int = 12

# 搜索规模守卫（防止无约束大分子枚举挂死）：
# - MAX_SEARCH_SECONDS：墙钟时间兜底（主守卫），超过即报错并给出引导；
# - MAX_SEARCH_NODES：节点预算（次守卫），防止单节点极慢的异常情况。
# 校准：C8H8O2（10 重原子）无约束约 23s 可通过；C9/C10 无约束在约 30s 内报错。
MAX_SEARCH_NODES: int = 1_500_000
MAX_SEARCH_SECONDS: float = 30.0

# 单价元素：隐氢模型无法承载氢（如 HF/HCl），此类分子退化为仅输入本身
_MONOVALENT: frozenset[str] = frozenset({"f", "cl", "br", "i"})

# ---- 基团约束（名称与 oc_features 注册表保持一致）----
_GROUP_CARBOXYL: str = "羧基"
_GROUP_NITRO: str = "硝基"
_GROUP_RING: str = "苯环"
_GROUP_ALDEHYDE: str = "醛基"
_GROUP_ESTER: str = "酯基"
_GROUP_AMIDE: str = "酰胺键"
_GROUP_KETONE: str = "酮羰基"
_GROUP_PHENOL_OH: str = "酚羟基"
_GROUP_AMINO: str = "氨基"
_GROUP_ETHER: str = "醚键"
_GROUP_ALCOHOL_OH: str = "醇羟基"
_GROUP_HALO: str = "卤代烃"
_GROUP_DOUBLE: str = "碳碳双键"
_GROUP_TRIPLE: str = "碳碳三键"
_GROUP_HYDROXY_ALIAS: str = "羟基"


class _NodeBudget:
    """搜索规模守卫：节点数或墙钟时间超限抛 ValueError。"""

    def __init__(self, limit: int, seconds: float) -> None:
        self.limit: int = limit
        self.seconds: float = seconds
        self.count: int = 0
        self._start: float = time.monotonic()

    def visit(self) -> None:
        self.count += 1
        if self.count > self.limit or (
            self.count & 4095 == 0
            and time.monotonic() - self._start > self.seconds
        ):
            raise ValueError(
                f"异构体搜索空间过大（已访问 {self.count} 个搜索节点，超过上限 "
                f"{self.limit} 或 {self.seconds:.0f} 秒）。建议：改用基团约束"
                "（required_groups，如['酯基']）缩小范围，或减小分子规模；"
                "如需放宽可调大 oc_isomers.MAX_SEARCH_SECONDS / MAX_SEARCH_NODES。"
            )


@dataclass(frozen=True)
class _FragmentSpec:
    """预建片段：固定内部结构 + 原子属性（元素/价键上限/基础占用/隐氢基数）。"""

    name: str
    atoms: tuple[tuple[str, int, int, int], ...]
    bonds: tuple[tuple[int, int, int], ...]
    dbe: int
    min_external: tuple[int, ...] = ()


# 可片段化的多原子基团（"至少含 1 个"语义下每个片段 k=1 即完备）：
# 羧基 / 酯基 / 酰胺键 = C=O + 桥接原子（DBE+1）；
# 醛基与酮羰基共用 C=O 片段（区别由后过滤判定）；双键 DBE+1；三键 DBE+2。
_FRAGMENT_SPECS: dict[str, _FragmentSpec] = {
    _GROUP_CARBOXYL: _FragmentSpec(
        _GROUP_CARBOXYL,
        (("c", 4, 3, 1), ("o", 2, 2, 0), ("o", 2, 1, 1)),
        ((0, 1, 2), (0, 2, 1)),
        1,
    ),
    _GROUP_ESTER: _FragmentSpec(
        _GROUP_ESTER,
        (("c", 4, 3, 1), ("o", 2, 2, 0), ("o", 2, 1, 1)),
        ((0, 1, 2), (0, 2, 1)),
        1,
        # 羰基碳允许 0 个外部键（甲酸酯）；桥氧必须连外部碳，排除羧酸。
        (0, 0, 1),
    ),
    _GROUP_ALDEHYDE: _FragmentSpec(
        _GROUP_ALDEHYDE,
        (("c", 4, 2, 2), ("o", 2, 2, 0)),
        ((0, 1, 2),),
        1,
    ),
    _GROUP_KETONE: _FragmentSpec(
        _GROUP_KETONE,
        (("c", 4, 2, 2), ("o", 2, 2, 0)),
        ((0, 1, 2),),
        1,
    ),
    _GROUP_AMIDE: _FragmentSpec(
        _GROUP_AMIDE,
        (("c", 4, 3, 1), ("o", 2, 2, 0), ("n", 3, 1, 2)),
        ((0, 1, 2), (0, 2, 1)),
        1,
    ),
    _GROUP_DOUBLE: _FragmentSpec(
        _GROUP_DOUBLE,
        (("c", 4, 2, 2), ("c", 4, 2, 2)),
        ((0, 1, 2),),
        1,
    ),
    _GROUP_TRIPLE: _FragmentSpec(
        _GROUP_TRIPLE,
        (("c", 4, 3, 1), ("c", 4, 3, 1)),
        ((0, 1, 3),),
        2,
    ),
}

# 单原子基团：暂不片段化（对搜索空间收缩有限），保留谓词剪枝并受守卫保护
_SINGLE_ATOM_GROUPS: frozenset[str] = frozenset({
    _GROUP_ALCOHOL_OH, _GROUP_PHENOL_OH, _GROUP_HYDROXY_ALIAS,
    _GROUP_AMINO, _GROUP_ETHER, _GROUP_HALO,
})

# 公开的合法基团名称（含"羟基"别名），供调用方与后续 UI 选择
REQUIRED_GROUP_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys([spec.name for spec in oc_features.GROUP_SPECS] + [_GROUP_HYDROXY_ALIAS])
)

# 硬编码名称必须都在 oc_features 注册表中，防止手误
assert {spec.name for spec in oc_features.GROUP_SPECS}.issuperset({
    _GROUP_CARBOXYL, _GROUP_NITRO, _GROUP_RING, _GROUP_ALDEHYDE, _GROUP_ESTER,
    _GROUP_AMIDE, _GROUP_KETONE, _GROUP_PHENOL_OH, _GROUP_AMINO, _GROUP_ETHER,
    _GROUP_ALCOHOL_OH, _GROUP_HALO, _GROUP_DOUBLE, _GROUP_TRIPLE,
})


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


def _resolve_required_groups(required_groups: Sequence[str] | None) -> frozenset[str]:
    """校验并去重基团名称；未知名称抛 ValueError。别名"羟基"在判断时按任一处理。"""
    if required_groups is None:
        return frozenset()
    valid = set(REQUIRED_GROUP_NAMES)
    resolved: set[str] = set()
    for name in required_groups:
        if name not in valid:
            raise ValueError(
                f"未知基团名称：{name!r}，可选：{'、'.join(REQUIRED_GROUP_NAMES)}"
            )
        resolved.add(name)
    return frozenset(resolved)


def _group_present(names: set[str], name: str) -> bool:
    """基团名是否满足："羟基"别名 = 醇羟基或酚羟基任一出现。"""
    if name == _GROUP_HYDROXY_ALIAS:
        return _GROUP_ALCOHOL_OH in names or _GROUP_PHENOL_OH in names
    return name in names


def _formula_possible(
    resolved: frozenset[str],
    heavy: dict[str, int],
    dbe: int,
    target: dict[str, int],
) -> bool:
    """公式级必要条件检查：任一必需基团在分子式层面不可能出现时返回 False。"""
    carbon = heavy.get('c', 0)
    nitrogen = heavy.get('n', 0)
    oxygen = heavy.get('o', 0)
    hydrogen = target.get('h', 0)
    halogens = (
        heavy.get('f', 0)
        + heavy.get('cl', 0)
        + heavy.get('br', 0)
        + heavy.get('i', 0)
    )
    for name in resolved:
        if name == _GROUP_RING:
            ok = carbon >= 6 and dbe >= 4
        elif name == _GROUP_NITRO:
            ok = carbon >= 1 and nitrogen >= 1 and oxygen >= 2 and dbe >= 1
        elif name == _GROUP_CARBOXYL:
            ok = carbon >= 1 and oxygen >= 2 and dbe >= 1
        elif name == _GROUP_ALDEHYDE:
            ok = carbon >= 1 and oxygen >= 1 and hydrogen >= 1 and dbe >= 1
        elif name == _GROUP_ESTER:
            ok = carbon >= 2 and oxygen >= 2 and dbe >= 1
        elif name == _GROUP_AMIDE:
            ok = carbon >= 1 and nitrogen >= 1 and oxygen >= 1 and dbe >= 1
        elif name == _GROUP_KETONE:
            ok = carbon >= 3 and oxygen >= 1 and dbe >= 1
        elif name == _GROUP_PHENOL_OH:
            ok = carbon >= 6 and oxygen >= 1 and dbe >= 4
        elif name == _GROUP_AMINO:
            ok = carbon >= 1 and nitrogen >= 1 and hydrogen >= 1
        elif name == _GROUP_ETHER:
            ok = carbon >= 2 and oxygen >= 1
        elif name == _GROUP_ALCOHOL_OH:
            ok = carbon >= 1 and oxygen >= 1
        elif name == _GROUP_HALO:
            ok = carbon >= 1 and halogens >= 1
        elif name == _GROUP_DOUBLE:
            ok = carbon >= 2 and dbe >= 1
        elif name == _GROUP_TRIPLE:
            ok = carbon >= 2 and dbe >= 2
        elif name == _GROUP_HYDROXY_ALIAS:
            ok = carbon >= 1 and oxygen >= 1
        else:
            ok = True
        if not ok:
            return False
    return True


def _contains_groups(molecule: oc.Molecule, resolved: frozenset[str]) -> bool:
    """候选分子是否包含全部必需基团（复用 oc_features，最终准确保证）。"""
    names = {group.name for group in oc_features.functional_groups(molecule)}
    return all(_group_present(names, name) for name in resolved)


def _reduce_hydrogen_pattern(seq: Sequence[int]) -> tuple[int, ...]:
    """按最大公约数约分并降序规范等位氢模式；空序列表示无氢。"""
    values = tuple(int(value) for value in seq)
    if not values:
        return ()
    divisor = 0
    for value in values:
        divisor = math.gcd(divisor, value)
    return tuple(sorted((value // divisor for value in values), reverse=True))


def _resolve_equivalent_hydrogens(
    equivalent_hydrogens: Sequence[int] | None,
) -> tuple[int, ...] | None:
    """校验并约分等位氢期望模式；None 表示不加约束。"""
    if equivalent_hydrogens is None:
        return None
    values = list(equivalent_hydrogens)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("等位氢模式必须为正整数序列（允许空序列表示不含氢）")
    return _reduce_hydrogen_pattern(values)


def _h_multiset_fits(
    h_values: Sequence[int],
    class_sizes: Sequence[int],
) -> bool:
    """逐原子终态氢数能否按期望类大小分组（等位氢必要条件）。

    同一等价类内各原子终态氢数相同；类大小 = 组内原子数 × 氢数（如甲苯
    的 CH3 类是 1 个原子 × 3 H，邻位类是 2 个原子 × 1 H）。判定等价于：
    把每个类大小 s 分配到某个氢值 v（要求 s 是 v 的整数倍），该值的原子
    被消耗 s/v 个，最终每个氢值的原子恰好用完。
    """
    values = sorted({value for value in h_values if value > 0}, reverse=True)
    if not values:
        return not class_sizes
    remaining = {value: sum(1 for h in h_values if h == value) for value in values}
    if sum(value * count for value, count in remaining.items()) != sum(class_sizes):
        return False
    sizes = sorted(class_sizes, reverse=True)

    def backtrack(index: int) -> bool:
        if index == len(sizes):
            return True
        size = sizes[index]
        for value in values:
            if size % value != 0:
                continue
            need = size // value
            if remaining[value] >= need:
                remaining[value] -= need
                if backtrack(index + 1):
                    return True
                remaining[value] += need
        return False

    return backtrack(0)


def find_isomers(
    molecule: oc.Molecule,
    required_groups: Sequence[str] | None = None,
    equivalent_hydrogens: Sequence[int] | None = None,
    allow_extra_rings: bool | None = None,
) -> list[oc.Molecule]:
    """返回分子的全部同分异构体（含输入结构本身的隐氢形式）。

    - 结果按 (tuple(feature), 生成序) 稳定排序，已按结构判等去重；
    - required_groups 指定必须含有的基团名称（每种至少 1 个），
      未知名称抛 ValueError；分子式层面不可能时直接返回空列表；
    - equivalent_hydrogens 指定期望的等位氢模式（按公约数约分后比较），
      含非正整数抛 ValueError；
    - 重原子数超过 MAX_HEAVY_ATOMS 抛 ValueError；
    - 无法用隐氢模型表示的分子（如 H2、HF）退化为仅返回输入本身；
    - 可含苯环的分子式只枚举含苯环/硝基的结构族（高中口径）；
    - allow_extra_rings 控制是否允许片段外部额外成环：默认在可含苯环的
      分子式中禁止（官方题解口径），普通分子式保留环状异构体；True 恢复
      允许额外环的完整图论行为（仍受搜索规模守卫限制）。
    """
    resolved = _resolve_required_groups(required_groups)
    h_pattern = _resolve_equivalent_hydrogens(equivalent_hydrogens)
    molecule.validate()
    target = dict(molecule.formula)
    heavy = _heavy_counts(molecule)
    total_heavy = sum(heavy.values())
    if total_heavy > MAX_HEAVY_ATOMS:
        raise ValueError(f"重原子数 {total_heavy} 超过上限 {MAX_HEAVY_ATOMS}")

    dbe = _formula_dbe(target)
    if not _formula_possible(resolved, heavy, dbe, target):
        return []
    target_h = target.get('h', 0)
    if h_pattern is not None:
        if h_pattern:
            pattern_sum = sum(h_pattern)
            if target_h == 0 or target_h % pattern_sum != 0:
                return []
            scale = target_h // pattern_sum
            h_classes: tuple[int, ...] | None = tuple(
                scale * value for value in h_pattern
            )
        else:
            if target_h != 0:
                return []
            h_classes = ()
    else:
        h_classes = None

    fragments = tuple(
        spec for name, spec in _FRAGMENT_SPECS.items() if name in resolved
    )
    benzene_capable = heavy.get('c', 0) >= 6 and dbe >= 4
    extra_rings_allowed = (
        not benzene_capable if allow_extra_rings is None else allow_extra_rings
    )
    budget = _NodeBudget(MAX_SEARCH_NODES, MAX_SEARCH_SECONDS)
    frag_dbe = sum(spec.dbe for spec in fragments)
    collected: list[oc.Molecule] = []
    if total_heavy == 0 or (
        all(element in _MONOVALENT for element in heavy) and target.get('h', 0) > 0
    ):
        collected.append(oc.copy_molecule(molecule))
    else:
        nitro_min = 1 if _GROUP_NITRO in resolved else 0
        if benzene_capable:
            for k_b in range(1, heavy.get('c', 0) // 6 + 1):
                mode_budget = dbe - 4 * k_b - frag_dbe
                if mode_budget < 0:
                    continue
                max_nitro = min(
                    heavy.get('n', 0), heavy.get('o', 0) // 2, mode_budget
                )
                for k_n in range(nitro_min, max_nitro + 1):
                    remaining_budget = mode_budget - k_n
                    fragment_names = {spec.name for spec in fragments}
                    if not extra_rings_allowed and remaining_budget > 0:
                        # 无额外环时，剩余 DBE 只能来自三键或双键。按
                        # "含三键 / 无三键但含双键" 拆成互斥模式，避免在
                        # 全部键级上盲目分支。
                        triple_fragments = (
                            fragments
                            if _GROUP_TRIPLE in fragment_names
                            else fragments + (_FRAGMENT_SPECS[_GROUP_TRIPLE],)
                        )
                        _collect_mode(
                            k_b, k_n, triple_fragments, heavy, target, resolved,
                            h_classes, budget, collected, extra_rings_allowed,
                        )
                        if _GROUP_TRIPLE not in fragment_names:
                            double_fragments = (
                                fragments
                                if _GROUP_DOUBLE in fragment_names
                                else fragments + (_FRAGMENT_SPECS[_GROUP_DOUBLE],)
                            )
                            _collect_mode(
                                k_b, k_n, double_fragments, heavy, target, resolved,
                                h_classes, budget, collected, extra_rings_allowed,
                                forbid_triple=True,
                            )
                    else:
                        _collect_mode(
                            k_b, k_n, fragments, heavy, target, resolved, h_classes,
                            budget, collected, extra_rings_allowed,
                        )
            if not collected and not resolved:
                # 罕见：分子式满足可含苯环但不存在含苯环结构 → 回退普通模式（仅无约束）
                max_nitro = min(heavy.get('n', 0), heavy.get('o', 0) // 2, dbe - frag_dbe)
                for k_n in range(nitro_min, max_nitro + 1):
                    if k_n > 0 and heavy.get('c', 0) == 0:
                        continue
                    _collect_mode(
                        0, k_n, fragments, heavy, target, resolved, h_classes,
                        budget, collected, extra_rings_allowed,
                    )
        else:
            max_nitro = min(heavy.get('n', 0), heavy.get('o', 0) // 2, dbe - frag_dbe)
            for k_n in range(nitro_min, max_nitro + 1):
                if k_n > 0 and heavy.get('c', 0) == 0:
                    continue
                _collect_mode(
                    0, k_n, fragments, heavy, target, resolved, h_classes,
                    budget, collected, extra_rings_allowed,
                )

    buckets: dict[tuple[str, ...], list[oc.Molecule]] = {}
    results: list[oc.Molecule] = []
    for candidate in collected:
        key = tuple(candidate.feature)
        bucket = buckets.setdefault(key, [])
        if any(candidate == other for other in bucket):
            continue
        bucket.append(candidate)
        results.append(candidate)
    if resolved or h_pattern is not None:
        # 先去重再过滤：官能团/等位氢检测只对唯一结构运行
        results = [
            candidate for candidate in results
            if (not resolved or _contains_groups(candidate, resolved))
            and (
                h_pattern is None
                or _reduce_hydrogen_pattern(
                    candidate.equivalent_hydrogen_groups
                ) == h_pattern
            )
        ]
    results.sort(key=lambda item: tuple(item.feature))
    return results


def _collect_mode(
    k_b: int,
    k_n: int,
    fragments: tuple[_FragmentSpec, ...],
    heavy: dict[str, int],
    target: dict[str, int],
    resolved: frozenset[str],
    h_classes: tuple[int, ...] | None,
    budget: _NodeBudget,
    collected: list[oc.Molecule],
    allow_extra_rings: bool,
    forbid_triple: bool = False,
) -> None:
    """枚举含 k_b 个苯环、k_n 个硝基与 fragments 中各 1 个片段的所有候选。

    递归原子清单：苯环槽位（6k_b 个）→ 硝基 N（k_n 个）→ 片段原子 →
    按元素排序的剩余自由原子。苯环/硝基/片段内部结构固定，不参与成键递归。
    resolved 中未片段化的单原子基团仍走谓词剪枝；budget 为节点预算守卫。
    """
    elements: list[str] = []
    limits: list[int] = []
    base: list[int] = []
    h0: list[int] = []
    is_nitro: list[bool] = []
    is_slot: list[bool] = []
    is_free: list[bool] = []
    min_ext: list[int] = []
    ring_id: list[int] = []
    frag_starts: list[int] = []
    for index in range(6 * k_b):
        elements.append('c')
        limits.append(4)
        base.append(3)          # 2 根环单键 + 1 个 π 槽位
        h0.append(1)
        is_nitro.append(False)
        is_slot.append(True)
        is_free.append(False)
        min_ext.append(0)
        ring_id.append(index // 6)
    for _ in range(k_n):
        elements.append('n')
        limits.append(4)        # 硝基型 [N,O,O] 中的 N 允许 4 个槽位
        base.append(3)          # 2 根 N-O 单键 + 1 个 π 槽位
        h0.append(1)
        is_nitro.append(True)
        is_slot.append(False)
        is_free.append(False)
        min_ext.append(0)
        ring_id.append(-1)
    for spec in fragments:
        frag_starts.append(len(elements))
        for index, (element, limit, base_used, h_zero) in enumerate(spec.atoms):
            elements.append(element)
            limits.append(limit)
            base.append(base_used)
            h0.append(h_zero)
            is_nitro.append(False)
            is_slot.append(False)
            is_free.append(False)
            need = spec.min_external[index] if index < len(spec.min_external) else 0
            min_ext.append(need)
            ring_id.append(-1)
    remaining: dict[str, int] = {element: count for element, count in heavy.items()}
    remaining['c'] = remaining.get('c', 0) - 6 * k_b
    remaining['n'] = remaining.get('n', 0) - k_n
    remaining['o'] = remaining.get('o', 0) - 2 * k_n
    for spec in fragments:
        for element, _limit, _base_used, _h_zero in spec.atoms:
            remaining[element] = remaining.get(element, 0) - 1
    if any(count < 0 for count in remaining.values()):
        return  # 原子预算不足，跳过该模式
    for element in sorted(remaining):
        for _ in range(remaining[element]):
            elements.append(element)
            limit = oc.CHEMISTRY_BOND_DICT[element]  # type: ignore
            limits.append(limit)
            base.append(0)
            h0.append(limit if limit >= 2 else 0)
            is_nitro.append(False)
            is_slot.append(False)
            is_free.append(True)
            min_ext.append(0)
            ring_id.append(-1)
    n = len(elements)
    if n == 0:
        return
    target_h = target.get('h', 0)
    target_dbe = _formula_dbe(target)
    base_dbe = 4 * k_b + k_n + sum(spec.dbe for spec in fragments)

    used: list[int] = list(base)
    bond_count: list[int] = [0] * n
    bonds: list[tuple[int, int, int]] = []
    bond_dbe: int = 0
    partner: list[int] = [-1] * n
    pairs: list[tuple[int, int]] = [
        (a, b) for a in range(n) for b in range(a + 1, n)
    ]
    halogens: frozenset[str] = frozenset({"f", "cl", "br", "i"})

    # 片段外部连通性：用于在高中口径下尽早禁止额外环。苯环与片段内部
    # 的固定键先并入同一分量；递归键加入/回退时同步维护分量标记。
    component_id: list[int] = list(range(n))

    def merge_components(a: int, b: int) -> list[tuple[int, int]]:
        """合并两个分量，返回可精确回滚的 (原子, 原分量) 记录。"""
        old_id = component_id[b]
        new_id = component_id[a]
        if old_id == new_id:
            return []
        changed = [
            (index, old_id)
            for index, value in enumerate(component_id)
            if value == old_id
        ]
        for index, _old in changed:
            component_id[index] = new_id
        return changed

    for index in range(n):
        if not is_slot[index]:
            continue
        for other in range(index + 1, n):
            if is_slot[other] and ring_id[index] == ring_id[other]:
                merge_components(index, other)
    for start, spec in zip(frag_starts, fragments):
        for a, b, _order in spec.bonds:
            merge_components(start + a, start + b)

    def current_h() -> int:
        total = 0
        for index in range(n):
            total += max(0, h0[index] - (used[index] - base[index]))
        return total

    def graph_state() -> tuple[int, bool]:
        """返回 (当前已形成环数, 递归原子是否全部连通)。

        并查集初始并入苯环内部、片段内部键，再并入已分配的键；
        环数用于不饱和度剪枝，连通性用于完成态构建前的廉价预检。
        """
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
        for start, spec in zip(frag_starts, fragments):
            for a, b, _order in spec.bonds:
                parent[find(start + b)] = find(start + a)
        rings = 0
        for a, b, _order in bonds:
            root_a, root_b = find(a), find(b)
            if root_a == root_b:
                rings += 1
            else:
                parent[root_b] = root_a
        root = find(0)
        connected = all(find(i) == root for i in range(n))
        return rings, connected

    def build_leaf() -> None:
        # 完成态预检：当前键已固定，递归原子未全连通则不可能成合法分子
        if not allow_extra_rings:
            if not all(
                component_id[index] == component_id[0]
                for index in range(n)
            ):
                return
        elif not graph_state()[1]:
            return
        # 氢数预检：不再有成键空间，隐氢数已定，与目标不符则公式必不匹配
        if current_h() != target_h:
            return
        # 等位氢预检：逐原子终态氢数必须能按期望类大小分组成同值组
        if h_classes is not None and not _h_multiset_fits(
            [max(0, h0[i] - (used[i] - base[i])) for i in range(n)],
            h_classes,
        ):
            return
        # 片段最少外部键预检（如酯基桥氧必须连碳，羧酸型会被跳过）
        for index, need in enumerate(min_ext):
            if need > 0 and bond_count[index] < need:
                return
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
        fragment_atom_lists: list[list[oc.Atom]] = []
        for spec in fragments:
            frag_atoms = [
                oc.Atom(element, molecule)
                for element, _limit, _base_used, _h_zero in spec.atoms
            ]
            for a, b, order in spec.bonds:
                oc.add_bond(frag_atoms[a], frag_atoms[b], order)
            fragment_atom_lists.append(frag_atoms)
        free_atoms: list[oc.Atom] = []
        for element in sorted(remaining):
            for _ in range(remaining[element]):
                free_atoms.append(oc.Atom(element, molecule))
        atoms = (
            ring_atoms
            + [group[0] for group in nitro_groups]
            + [atom for frag in fragment_atom_lists for atom in frag]
            + free_atoms
        )
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

    # ---- 基团必要条件剪枝（仅在存在约束时启用；宁可少剪，不可剪错）----
    def bondable(a: int, b: int, caps: list[int]) -> bool:
        """未决原子对是否还能成键（排除同环槽位对、容量不足）。"""
        if is_slot[a] and is_slot[b] and ring_id[a] == ring_id[b]:
            return False
        return caps[a] >= 1 and caps[b] >= 1

    def has_carbon_neighbor(index: int) -> bool:
        """已分配的键中是否与碳相邻。"""
        for x, y, o in bonds:
            if o > 0:
                if x == index and elements[y] == 'c':
                    return True
                if y == index and elements[x] == 'c':
                    return True
        return False

    def pending_carbon_pair(
        index: int,
        caps: list[int],
        pending: list[tuple[int, int]],
    ) -> bool:
        """未决原子对中是否存在与碳的可成键对。"""
        for a, b in pending:
            if a == index:
                other = b
            elif b == index:
                other = a
            else:
                continue
            if elements[other] == 'c' and bondable(index, other, caps):
                return True
        return False

    def free_o_oh_possible(
        caps: list[int],
        pending: list[tuple[int, int]],
    ) -> bool:
        """存在可成为 O-H 的游离 O（当前占用 <=1，已连碳或可连碳）。"""
        for a in range(n):
            if not is_free[a] or elements[a] != 'o' or used[a] > 1:
                continue
            if has_carbon_neighbor(a) or pending_carbon_pair(a, caps, pending):
                return True
        return False

    def free_n_amino_possible(
        caps: list[int],
        pending: list[tuple[int, int]],
    ) -> bool:
        """存在可挂氢且连碳的游离 N（当前占用 <=2）。"""
        for a in range(n):
            if not is_free[a] or elements[a] != 'n' or used[a] > 2:
                continue
            if has_carbon_neighbor(a) or pending_carbon_pair(a, caps, pending):
                return True
        return False

    def ether_o_possible(
        caps: list[int],
        pending: list[tuple[int, int]],
    ) -> bool:
        """存在可连两个碳的 O（醚键 / 酯桥）。"""
        for a in range(n):
            if not is_free[a] or elements[a] != 'o' or used[a] > 2:
                continue
            c_bonds = 0
            for x, y, o in bonds:
                if o > 0:
                    if x == a and elements[y] == 'c':
                        c_bonds += 1
                    elif y == a and elements[x] == 'c':
                        c_bonds += 1
            c_pending = 0
            for a2, b2 in pending:
                if a2 == a:
                    other = b2
                elif b2 == a:
                    other = a2
                else:
                    continue
                if elements[other] == 'c' and bondable(a, other, caps):
                    c_pending += 1
            if c_bonds + c_pending >= 2:
                return True
        return False

    def phenol_o_possible(
        caps: list[int],
        pending: list[tuple[int, int]],
    ) -> bool:
        """存在可连苯环槽位的未饱和 O（酚羟基）。"""
        if k_b == 0:
            return False
        for a in range(n):
            if not is_free[a] or elements[a] != 'o' or used[a] > 1:
                continue
            slot_bonded = False
            for x, y, o in bonds:
                if o > 0 and a in (x, y):
                    slot_bonded = is_slot[y if x == a else x]
                    if slot_bonded:
                        break
            if slot_bonded:
                return True
            for a2, b2 in pending:
                if a2 == a:
                    other = b2
                elif b2 == a:
                    other = a2
                else:
                    continue
                if is_slot[other] and bondable(a, other, caps):
                    return True
        return False

    def halo_possible(
        caps: list[int],
        pending: list[tuple[int, int]],
    ) -> bool:
        """已存在或仍可形成 C-卤素 键。"""
        for a, b, o in bonds:
            if o > 0 and (
                (elements[a] == 'c' and elements[b] in halogens)
                or (elements[b] == 'c' and elements[a] in halogens)
            ):
                return True
        for a, b in pending:
            if bondable(a, b, caps) and (
                (elements[a] == 'c' and elements[b] in halogens)
                or (elements[b] == 'c' and elements[a] in halogens)
            ):
                return True
        return False

    def feasible(pos: int) -> bool:
        """所有必需基团在当前状态下仍有可能形成（必要条件，保守）。"""
        if not resolved:
            return True
        caps = [limits[i] - used[i] for i in range(n)]
        pending = pairs[pos:]
        for name in resolved:
            if name == _GROUP_RING:
                if k_b == 0:
                    return False
            elif name == _GROUP_NITRO:
                if k_n == 0:
                    return False
            elif name == _GROUP_ALCOHOL_OH:
                if not free_o_oh_possible(caps, pending):
                    return False
            elif name == _GROUP_PHENOL_OH:
                if not phenol_o_possible(caps, pending):
                    return False
            elif name == _GROUP_HYDROXY_ALIAS:
                if not (
                    free_o_oh_possible(caps, pending)
                    or phenol_o_possible(caps, pending)
                ):
                    return False
            elif name == _GROUP_AMINO:
                if not free_n_amino_possible(caps, pending):
                    return False
            elif name == _GROUP_ETHER:
                if not ether_o_possible(caps, pending):
                    return False
            elif name == _GROUP_HALO:
                if not halo_possible(caps, pending):
                    return False
        return True

    def backtrack(i: int, j: int, pos: int) -> None:
        nonlocal bond_dbe
        budget.visit()
        if i == n:
            build_leaf()
            return
        h_now = current_h()
        if h_now < target_h:
            return
        if h_now == target_h:
            # 再成键只会减少氢数：剩余原子对只能全为 0，直接按当前状态收尾
            if not feasible(pos):
                return
            build_leaf()
            return
        if j == n:
            # 原子 i 的全部对外键已确定：剪枝孤立自由原子 / 硝基 N 缺挂载
            if is_nitro[i]:
                if partner[i] < 0 or elements[partner[i]] != 'c':
                    return
            elif is_free[i] and n > 1 and bond_count[i] == 0:
                return
            elif min_ext[i] > 0 and bond_count[i] < min_ext[i]:
                return
            backtrack(i + 1, i + 2, pos)
            return
        if allow_extra_rings:
            if base_dbe + bond_dbe + graph_state()[0] > target_dbe:
                return
        elif base_dbe + bond_dbe > target_dbe:
            return
        if not feasible(pos):
            return
        if is_slot[i] and is_slot[j] and ring_id[i] == ring_id[j]:
            # 同一苯环的两个槽位之间禁止成键（保持苯环为干净的单环）
            backtrack(i, j + 1, pos + 1)
            return
        max_order = min(
            2 if forbid_triple else 3,
            limits[i] - used[i],
            limits[j] - used[j],
        )
        for order in range(max_order + 1):
            if order > 0:
                if not allow_extra_rings and component_id[i] == component_id[j]:
                    # 该键会闭合片段外部新环；高中口径下直接剪掉。
                    continue
                merged = merge_components(i, j)
                used[i] += order
                used[j] += order
                bond_count[i] += 1
                bond_count[j] += 1
                bonds.append((i, j, order))
                bond_dbe += order - 1
                if is_nitro[i]:
                    partner[i] = j
                if is_nitro[j]:
                    partner[j] = i
            backtrack(i, j + 1, pos + 1)
            if order > 0:
                bonds.pop()
                bond_count[i] -= 1
                bond_count[j] -= 1
                bond_dbe -= order - 1
                used[i] -= order
                used[j] -= order
                for index, old_id in merged:
                    component_id[index] = old_id
                if is_nitro[i]:
                    partner[i] = -1
                if is_nitro[j]:
                    partner[j] = -1

    backtrack(0, 1, 0)


def isomer_report(
    molecule: oc.Molecule,
    required_groups: Sequence[str] | None = None,
    equivalent_hydrogens: Sequence[int] | None = None,
) -> str:
    """控制台文字报告：分子式、基团约束、总数、每个异构体的编号/分子式/不饱和度/环数。"""
    isomers = find_isomers(
        molecule, required_groups, equivalent_hydrogens=equivalent_hydrogens
    )
    lines: list[str] = [f"分子式：{_display_formula(molecule.formula)}"]
    if required_groups:
        lines.append(f"基团约束：{'、'.join(dict.fromkeys(required_groups))}")
    lines.append(f"同分异构体总数：{len(isomers)}")
    lines.append("")
    for index, candidate in enumerate(isomers, 1):
        lines.append(
            f"异构体 {index:02d}：{_display_formula(candidate.formula)}，"
            f"不饱和度 {candidate.unsaturation}，环数 {candidate.ring_count}"
        )
    return "\n".join(lines)


def save_isomers(
    molecule: oc.Molecule,
    directory: str | os.PathLike[str],
    required_groups: Sequence[str] | None = None,
    equivalent_hydrogens: Sequence[int] | None = None,
) -> list[Path]:
    """把全部异构体保存为构建脚本（oc_io），返回文件路径列表。

    目录结构：<directory>/<公式键>/isomer_01.py、isomer_02.py...
    """
    isomers = find_isomers(
        molecule, required_groups, equivalent_hydrogens=equivalent_hydrogens
    )
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
    print(isomer_report(demo, required_groups=["醇羟基"]))
