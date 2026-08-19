"""基团与活性位点分析（无第三方依赖）。

设计：
- 基团 = 命名子结构模式（复用 organic_chemistry 的 Pattern 匹配器），
  全部为派生分析：不改动分子结构，不参与指纹/判等，不需要序列化。
- 重叠抑制：具体基团优先（priority 越大越具体），core 特征原子被已保留的
  异名基团占据时丢弃；同基团多次出现（如乙二醇两个羟基、CHCl3 三个卤代键）
  各返回一个实例，同一基团在同一位置（core 原子集合相同）的重复匹配合并。
- 活性位点：在基团结果之上按规则生成原子级标注；隐氢位点锚定在承载氢的
  重原子（碳/氧）上，仅 ActiveH 显式节点存在时酸性位点锚定到该节点。
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
    """

    def __init__(self, atom: oc.Atom, kind: SiteKind, label: str) -> None:
        self.atom: oc.Atom = atom
        self.kind: SiteKind = kind
        self.label: str = label

    def __repr__(self) -> str:
        return f"<ActiveSite {self.kind} {self.atom} {self.label}>"


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
    具体基团（priority 高）优先保留并抑制被其覆盖的通用基团。
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
    for spec, match in raw:
        core_atoms = [match.atom_map[p_atom] for p_atom in spec.core]
        if any(atom in covered and covered[atom] != spec.name for atom in core_atoms):
            continue
        for atom in core_atoms:
            covered.setdefault(atom, spec.name)
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


def active_sites(molecule: oc.Molecule) -> list[ActiveSite]:
    """返回分子中的活性位点列表（派生分析）。

    在基团检测结果之上按规则生成；同一 (原子, 位点类型) 只保留一个。
    """
    groups = functional_groups(molecule)
    sites: list[ActiveSite] = []
    seen: set[tuple[oc.Atom, SiteKind]] = set()

    def add(atom: oc.Atom, kind: SiteKind, label: str) -> None:
        key = (atom, kind)
        if key in seen:
            return
        seen.add(key)
        sites.append(ActiveSite(atom, kind, label))

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
            for atom in group.atoms:
                if _total_h(atom) >= 1:
                    add(atom, 'substitution', '苯环的取代位点')
        if group.name == '卤代烃':
            c_atom = next((atom for atom in group.atoms if atom.name == 'c'), None)
            if c_atom is not None:
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
