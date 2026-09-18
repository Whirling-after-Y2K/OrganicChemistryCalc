"""基团与活性位点分析（无第三方依赖）。

设计：
- 基团 = 命名子结构模式（复用 organic_chemistry 的 Pattern 匹配器），
  全部为派生分析：不改动分子结构，不参与指纹/判等，不需要序列化。
- 重叠抑制：具体基团优先（priority 越大越具体），core 特征原子被已保留的
  异名基团占据时丢弃；同基团多次出现（如乙二醇两个羟基、CHCl3 三个卤代键）
  各返回一个实例，同一基团在同一位置（core 原子集合相同）的重复匹配合并。
- 活性位点：在基团结果之上按规则生成原子级标注；隐氢位点锚定在承载氢的
  重原子（碳/氧）上，仅 ActiveH 显式节点存在时酸性位点锚定到该节点。
- 苯环取代位点：结合定位效应只标有利位置（邻对位/间位），多取代时按每个
  取代基分别标注；ActiveSite.position 记录 邻位/间位/对位。
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import organic_chemistry as oc


SiteKind: TypeAlias = Literal[
    "alpha_h",
    "acidic_h",
    "addition",
    "substitution",
    "oxidation",
    "reduction",
    "hydrolysis",
]


class GroupSpec:
    """基团规格：命名模式 + 重叠抑制所需的核心原子与优先级。"""

    def __init__(
        self,
        name: str,
        category: str,
        pattern: oc.Pattern,
        core: tuple[oc.PatternAtom, ...],
        priority: int,
    ) -> None:
        self.name: str = name
        self.category: str = category
        self.pattern: oc.Pattern = pattern
        self.core: tuple[oc.PatternAtom, ...] = core
        self.priority: int = priority

    def __repr__(self) -> str:
        return f"<GroupSpec {self.name} priority={self.priority}>"


class FunctionalGroup:
    """分子中检测到的一个基团。

    atoms: 命中的特征原子（按 core 顺序）；anchor: 基团中第一个与环外原子
    成键的原子（便于 GUI 定位），无环外键时为 None。
    """

    def __init__(
        self,
        name: str,
        category: str,
        atoms: tuple[oc.Atom, ...],
        anchor: oc.Atom | None,
    ) -> None:
        self.name: str = name
        self.category: str = category
        self.atoms: tuple[oc.Atom, ...] = atoms
        self.anchor: oc.Atom | None = anchor

    def __repr__(self) -> str:
        names = ','.join(atom.name for atom in self.atoms)
        return f"<FunctionalGroup {self.name} ({names})>"


class ActiveSite:
    """原子级活性位点标注。

    kind 取值：alpha_h 羧基/醛基/酮羰基的 α-H；acidic_h 羧基/酚羟基的酸性氢；
    addition 碳碳双键/三键的加成位点；substitution 苯环/卤代烃的取代位点；
    oxidation 醇/醛的氧化位点；reduction 硝基的还原位点；hydrolysis 酯的水解位点。
    position 为苯环取代位点的几何位置（邻位/间位/对位，多取代时合并），
    非苯环位点为 None。
    """

    def __init__(
        self,
        atom: oc.Atom,
        kind: SiteKind,
        label: str,
        position: str | None = None,
    ) -> None:
        self.atom: oc.Atom = atom
        self.kind: SiteKind = kind
        self.label: str = label
        self.position: str | None = position

    def __repr__(self) -> str:
        suffix = f" [{self.position}]" if self.position is not None else ""
        return f"<ActiveSite {self.kind} {self.atom} {self.label}{suffix}>"


def _total_h(atom: oc.Atom) -> int:
    """原子的总氢数 = 隐氢 + 显式 H 邻居数（ActiveH 计入）。"""
    explicit = sum(1 for bond in atom.bonds if bond.other(atom).name == 'h')
    return atom.implicit_h + explicit


def _h_neighbors(atom: oc.Atom) -> list[oc.Atom]:
    """显式 H 邻居列表（普通显式 H 与 ActiveH）。"""
    return [bond.other(atom) for bond in atom.bonds if bond.other(atom).name == 'h']


def _build_group_specs() -> tuple[GroupSpec, ...]:
    """构建基团注册表（14 种基团、17 个规格，卤代烃按 F/Cl/Br/I 拆分）。"""
    specs: list[GroupSpec] = []

    # ---- priority 100：最具体，优先保留 ----
    p = oc.Pattern()
    c_carbonyl = p.add_atom('c')
    o_double = p.add_atom('o')
    o_h = p.add_atom('o', h_min=1)
    p.add_bond(c_carbonyl, o_double, order=2)
    p.add_bond(c_carbonyl, o_h)
    specs.append(GroupSpec('羧基', '含氧', p, (c_carbonyl, o_double, o_h), 100))

    p = oc.Pattern()
    n = p.add_atom('n', pi=True, pi_group=1)
    o1 = p.add_atom('o', pi=True, pi_group=1)
    o2 = p.add_atom('o', pi=True, pi_group=1)
    p.add_bond(n, o1)
    p.add_bond(n, o2)
    specs.append(GroupSpec('硝基', '含氮', p, (n, o1, o2), 100))

    p = oc.Pattern()
    ring: list[oc.PatternAtom] = [
        p.add_atom('c', pi=True, pi_group=1) for _ in range(6)
    ]
    for i in range(6):
        p.add_bond(ring[i], ring[(i + 1) % 6])
    specs.append(GroupSpec('苯环', '芳环', p, tuple(ring), 100))

    # ---- priority 90 ----
    p = oc.Pattern()
    c_carbonyl = p.add_atom('c', h_min=1)
    o_double = p.add_atom('o')
    p.add_bond(c_carbonyl, o_double, order=2)
    specs.append(GroupSpec('醛基', '含氧', p, (c_carbonyl, o_double), 90))

    p = oc.Pattern()
    c_carbonyl = p.add_atom('c')
    o_double = p.add_atom('o')
    o_bridge = p.add_atom('o')
    c_rest = p.add_atom('c')
    p.add_bond(c_carbonyl, o_double, order=2)
    p.add_bond(c_carbonyl, o_bridge)
    p.add_bond(o_bridge, c_rest)
    specs.append(GroupSpec('酯基', '含氧', p, (c_carbonyl, o_double, o_bridge), 90))

    p = oc.Pattern()
    c_carbonyl = p.add_atom('c')
    o_double = p.add_atom('o')
    n = p.add_atom('n', h_min=1)
    p.add_bond(c_carbonyl, o_double, order=2)
    p.add_bond(c_carbonyl, n)
    specs.append(GroupSpec('酰胺键', '含氮', p, (c_carbonyl, o_double, n), 90))

    # ---- priority 80 ----
    p = oc.Pattern()
    c_carbonyl = p.add_atom('c', h=0)
    o_double = p.add_atom('o')
    neighbor = p.add_atom()  # 至少一个单键邻居，排除 CO2
    p.add_bond(c_carbonyl, o_double, order=2)
    p.add_bond(c_carbonyl, neighbor, order=1)
    specs.append(GroupSpec('酮羰基', '含氧', p, (c_carbonyl, o_double), 80))

    p = oc.Pattern()
    o = p.add_atom('o', h_min=1, pi=False)
    c = p.add_atom('c', pi=True)
    p.add_bond(c, o)
    specs.append(GroupSpec('酚羟基', '含氧', p, (o,), 80))

    p = oc.Pattern()
    n = p.add_atom('n', h_min=1)
    c = p.add_atom('c')
    p.add_bond(n, c)
    specs.append(GroupSpec('氨基', '含氮', p, (n,), 80))

    p = oc.Pattern()
    o = p.add_atom('o')
    c1 = p.add_atom('c')
    c2 = p.add_atom('c')
    p.add_bond(o, c1)
    p.add_bond(o, c2)
    specs.append(GroupSpec('醚键', '含氧', p, (o,), 80))

    # ---- priority 70 ----
    p = oc.Pattern()
    o = p.add_atom('o', h_min=1, pi=False)
    c = p.add_atom('c', pi=False)
    p.add_bond(c, o)
    specs.append(GroupSpec('醇羟基', '含氧', p, (o,), 70))

    for halogen in ('f', 'cl', 'br', 'i'):
        p = oc.Pattern()
        c = p.add_atom('c')
        x = p.add_atom(halogen)
        p.add_bond(c, x)
        specs.append(GroupSpec('卤代烃', '卤代', p, (c, x), 70))

    p = oc.Pattern()
    c1 = p.add_atom('c')
    c2 = p.add_atom('c')
    p.add_bond(c1, c2, order=2)
    specs.append(GroupSpec('碳碳双键', '烃类', p, (c1, c2), 70))

    p = oc.Pattern()
    c1 = p.add_atom('c')
    c2 = p.add_atom('c')
    p.add_bond(c1, c2, order=3)
    specs.append(GroupSpec('碳碳三键', '烃类', p, (c1, c2), 70))

    return tuple(specs)


GROUP_SPECS: tuple[GroupSpec, ...] = _build_group_specs()


def _compute_anchor(atoms: tuple[oc.Atom, ...]) -> oc.Atom | None:
    """锚点 = 基团中第一个与环外原子成键的原子；无环外键时为 None。"""
    group_set = set(atoms)
    for atom in atoms:
        for bond in atom.bonds:
            if bond.other(atom) not in group_set:
                return atom
    return None


def functional_groups(molecule: oc.Molecule) -> list[FunctionalGroup]:
    """返回分子中检测到的基团列表（派生分析，按具体程度排序）。

    检测前先校验分子结构，无效结构抛 ValueError。同一基团在同一位置
    （core 原子集合相同）的重复匹配合并为一次；异名基团重叠时，
    仅严格更高优先级的基团抑制低优先级基团；同优先级允许共存
    （如甲酸酯同时含有酯基与醛基）。
    """
    molecule.validate()
    raw: list[tuple[GroupSpec, oc.SubstructureMatch]] = []
    for spec in GROUP_SPECS:
        seen_cores: set[frozenset[oc.Atom]] = set()
        for match in oc.find_substructure_matches(spec.pattern, molecule):
            core = frozenset(match.atom_map[p_atom] for p_atom in spec.core)
            if core in seen_cores:
                continue
            seen_cores.add(core)
            raw.append((spec, match))
    raw.sort(key=lambda item: -item[0].priority)

    kept: list[tuple[GroupSpec, oc.SubstructureMatch]] = []
    covered: dict[oc.Atom, str] = {}
    covered_priority: dict[oc.Atom, int] = {}
    for spec, match in raw:
        core_atoms = [match.atom_map[p_atom] for p_atom in spec.core]
        # 苯环占据的环碳不抑制取代基类基团（如氯苯的 卤代烃 与 苯环 共存）；
        # 同优先级不互斥，保证甲酸酯的 醛基/酯基 共同保留。
        if any(
            atom in covered
            and covered[atom] != spec.name
            and covered[atom] != '苯环'
            and covered_priority[atom] > spec.priority
            for atom in core_atoms
        ):
            continue
        for atom in core_atoms:
            covered.setdefault(atom, spec.name)
            covered_priority.setdefault(atom, spec.priority)
        kept.append((spec, match))

    groups: list[FunctionalGroup] = []
    for spec, match in kept:
        atoms = tuple(match.atom_map[p_atom] for p_atom in spec.core)
        groups.append(FunctionalGroup(spec.name, spec.category, atoms, _compute_anchor(atoms)))
    return groups


def has_group(molecule: oc.Molecule, name: str) -> bool:
    """分子是否含有指定名称的基团。"""
    return any(group.name == name for group in functional_groups(molecule))


def _carbonyl_carbon(group: FunctionalGroup) -> oc.Atom | None:
    """基团中带 C=O（与组内 O 成键级 2 键）的碳原子。"""
    for atom in group.atoms:
        for bond in atom.bonds:
            other = bond.other(atom)
            if other in group.atoms and other.name == 'o' and bond.order == 2:
                return atom
    return None


def _alpha_carbons(carbonyl: oc.Atom) -> list[oc.Atom]:
    """与羰基碳直接相连且带氢的碳（α 碳）。"""
    result: list[oc.Atom] = []
    for bond in carbonyl.bonds:
        neighbor = bond.other(carbonyl)
        if neighbor.name == 'c' and _total_h(neighbor) >= 1:
            result.append(neighbor)
    return result


def _acidic_anchor(o_atom: oc.Atom) -> oc.Atom:
    """酸性氢位点锚点：O 上挂了 ActiveH 时用该节点，否则用 O 本身。"""
    for neighbor in _h_neighbors(o_atom):
        if isinstance(neighbor, oc.ActiveH):
            return neighbor
    return o_atom


Directing: TypeAlias = Literal["ortho_para", "meta"]

_HALOGEN_NAMES: dict[str, str] = {"f": "氟", "cl": "氯", "br": "溴", "i": "碘"}
_ORTHO_PARA_GROUPS: frozenset[str] = frozenset({"酚羟基", "氨基", "醚键", "卤代烃", "苯环"})
_META_GROUPS: frozenset[str] = frozenset({"硝基", "羧基", "醛基", "酯基", "酮羰基", "酰胺键"})


def _directing_of(substituent: str) -> Directing:
    """按取代基名称返回定位类型：邻对位定位基 或 间位定位基。"""
    if substituent in _ORTHO_PARA_GROUPS or substituent == '烷基':
        return 'ortho_para'
    return 'meta'


def _ring_cycle(ring_atoms: tuple[oc.Atom, ...]) -> list[oc.Atom] | None:
    """把 6 个环碳按成键顺序排成一圈；无法成环时返回 None。"""
    ring_set = set(ring_atoms)
    if len(ring_set) != 6:
        return None
    start = ring_atoms[0]
    ring_neighbors = [
        bond.other(start) for bond in start.bonds
        if bond.other(start) in ring_set
    ]
    if len(ring_neighbors) != 2:
        return None
    cycle: list[oc.Atom] = [start]
    prev, current = start, ring_neighbors[0]
    while current is not start:
        cycle.append(current)
        next_atom: oc.Atom | None = None
        for bond in current.bonds:
            candidate = bond.other(current)
            if candidate in ring_set and candidate is not prev:
                next_atom = candidate
                break
        if next_atom is None:
            return None
        prev, current = current, next_atom
        if len(cycle) > 6:
            return None
    return cycle if len(cycle) == 6 else None


def _attached_groups(
    atom: oc.Atom,
    ring_set: set[oc.Atom],
    groups: list[FunctionalGroup],
) -> list[FunctionalGroup]:
    """挂在该原子上的基团：基团中存在"不属于本环"的原子与该原子成键。"""
    result: list[FunctionalGroup] = []
    for group in groups:
        for group_atom in group.atoms:
            if group_atom in ring_set:
                continue
            if any(bond.other(group_atom) is atom for bond in group_atom.bonds):
                result.append(group)
                break
    return result


def _substituent_info(
    ring_atom: oc.Atom,
    ring_set: set[oc.Atom],
    groups: list[FunctionalGroup],
) -> tuple[str, Directing] | None:
    """返回 (取代基标签, 定位类型)；无法识别时返回 None。

    优先用挂在该环碳上的已保留基团；无基团时按外部邻居原子兜底
    （卤素、烷基、羟基、氨基均视为邻对位定位基）。
    """
    for group in _attached_groups(ring_atom, ring_set, groups):
        if group.name == '苯环':
            return '苯基', 'ortho_para'
        if group.name == '卤代烃':
            halogen = next(
                (atom for atom in group.atoms if atom.name in _HALOGEN_NAMES),
                None,
            )
            if halogen is not None:
                return _HALOGEN_NAMES[halogen.name], 'ortho_para'
            continue
        if group.name in _ORTHO_PARA_GROUPS or group.name in _META_GROUPS:
            return group.name, _directing_of(group.name)
    for bond in ring_atom.bonds:
        neighbor = bond.other(ring_atom)
        if neighbor in ring_set or neighbor.name == 'h':
            continue
        if neighbor.name in _HALOGEN_NAMES:
            return _HALOGEN_NAMES[neighbor.name], 'ortho_para'
        if neighbor.name == 'c':
            return '烷基', 'ortho_para'
        if neighbor.name == 'o':
            return '羟基', 'ortho_para'
        if neighbor.name == 'n':
            return '氨基', 'ortho_para'
    return None


def _in_pi_system(atom: oc.Atom) -> bool:
    """原子是否属于某个 π 体系。"""
    return any(atom in pi.atoms for pi in atom.belong.pi_systems)


def active_sites(molecule: oc.Molecule) -> list[ActiveSite]:
    """返回分子中的活性位点列表（派生分析）。

    在基团检测结果之上按规则生成；同一 (原子, 位点类型) 只保留一个。
    """
    groups = functional_groups(molecule)
    sites: list[ActiveSite] = []
    seen: set[tuple[oc.Atom, SiteKind]] = set()

    def add(
        atom: oc.Atom,
        kind: SiteKind,
        label: str,
        position: str | None = None,
    ) -> None:
        key = (atom, kind)
        if key in seen:
            return
        seen.add(key)
        sites.append(ActiveSite(atom, kind, label, position=position))

    for group in groups:
        if group.name in ('羧基', '醛基', '酮羰基'):
            carbonyl = _carbonyl_carbon(group)
            if carbonyl is not None:
                for alpha in _alpha_carbons(carbonyl):
                    add(alpha, 'alpha_h', f"{group.name}的 α-H（{_total_h(alpha)} 个）")
        if group.name in ('羧基', '酚羟基'):
            for atom in group.atoms:
                if atom.name == 'o' and _total_h(atom) >= 1:
                    add(_acidic_anchor(atom), 'acidic_h', f"{group.name}的酸性氢")
                    break
        if group.name in ('碳碳双键', '碳碳三键'):
            for atom in group.atoms:
                add(atom, 'addition', f"{group.name}的加成位点")
        if group.name == '苯环':
            ring_set = set(group.atoms)
            other_groups = [g for g in groups if g is not group]
            cycle = _ring_cycle(group.atoms)
            if cycle is None:
                for atom in group.atoms:
                    if _total_h(atom) >= 1:
                        add(atom, 'substitution', '苯环的取代位点')
                continue
            substituents: dict[int, tuple[str, Directing]] = {}
            for index, ring_atom in enumerate(cycle):
                info = _substituent_info(ring_atom, ring_set, other_groups)
                if info is not None:
                    substituents[index] = info
            if not substituents:
                for atom in group.atoms:
                    if _total_h(atom) >= 1:
                        add(atom, 'substitution', '苯环的取代位点')
                continue
            for index, ring_atom in enumerate(cycle):
                if _total_h(ring_atom) < 1:
                    continue
                annotations: list[tuple[str, str]] = []
                for s_index, (s_label, s_directing) in substituents.items():
                    if s_index == index:
                        continue
                    distance = min((index - s_index) % 6, (s_index - index) % 6)
                    if distance == 1:
                        position = '邻位'
                    elif distance == 2:
                        position = '间位'
                    else:
                        position = '对位'
                    favored = (
                        position in ('邻位', '对位')
                        if s_directing == 'ortho_para'
                        else position == '间位'
                    )
                    if favored:
                        annotations.append((position, s_label))
                if not annotations:
                    continue
                if len(annotations) == 1:
                    position, s_label = annotations[0]
                    add(
                        ring_atom,
                        'substitution',
                        f'苯环的{position}取代位点（{s_label}）',
                        position=position,
                    )
                else:
                    positions = '、'.join(dict.fromkeys(p for p, _ in annotations))
                    details = '；'.join(f'{p}·{s}' for p, s in annotations)
                    add(
                        ring_atom,
                        'substitution',
                        f'苯环的取代位点（{details}）',
                        position=positions,
                    )
        if group.name == '卤代烃':
            c_atom = next((atom for atom in group.atoms if atom.name == 'c'), None)
            # 芳香环上的 C-X 不进行亲核取代/消去（如氯苯），不标取代位点
            if c_atom is not None and not _in_pi_system(c_atom):
                add(c_atom, 'substitution', '卤代烃的取代位点')
        if group.name == '醇羟基':
            o_atom = next((atom for atom in group.atoms if atom.name == 'o'), None)
            if o_atom is not None:
                c_atom = next(
                    (bond.other(o_atom) for bond in o_atom.bonds if bond.other(o_atom).name == 'c'),
                    None,
                )
                if c_atom is not None:
                    add(c_atom, 'oxidation', '醇羟基的氧化位点')
        if group.name == '醛基':
            carbonyl = _carbonyl_carbon(group)
            if carbonyl is not None:
                add(carbonyl, 'oxidation', '醛基的氧化位点')
        if group.name == '硝基':
            n_atom = next((atom for atom in group.atoms if atom.name == 'n'), None)
            if n_atom is not None:
                add(n_atom, 'reduction', '硝基的还原位点')
        if group.name == '酯基':
            carbonyl = _carbonyl_carbon(group)
            if carbonyl is not None:
                add(carbonyl, 'hydrolysis', '酯基的水解位点')
    return sites
