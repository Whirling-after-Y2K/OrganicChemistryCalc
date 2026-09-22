"""有机反应模拟模块（无第三方依赖）。

设计：
- 反应规则 = 反应物子结构模式（复用 organic_chemistry.Pattern）或整分子式
  + 计量数 + 变换函数。规则声明每个输入反应物的规格（ReactantSpec）：
  pattern 用于有机分子位点匹配，formula 用于 H2/X2/H2O/O2/HNO3 等小分子
  整分子匹配；count 为化学计量数（引擎按计量数复制输入分子，用户每种
  反应物只需输入一次）。
- 引擎流程：按规格复制输入分子 → 对每份副本枚举子结构匹配 → 全部副本的
  位点做笛卡尔积 → 用 merge_molecules 合并进一个工作分子 → 执行变换 →
  校验 → 按连通分量拆分主产物 → 追加规则显式构造的副产物 → 结构判等去重。
- 隐氢约定：产物氢优先以隐氢自动平衡（自由价变化自动调整）；必须显式进入
  副产物的 H/X 由规则显式构造，并删除不再属于产物的游离原子。
- 元素限制：元素表仅 C/N/O/H/F/Cl/Br/I。涉及 Na/K/Cu/Ag/Mn/S 等的反应以
   可表示的等价形式表达（如卤代烃水解写 R-X + H2O → R-OH + HX，其逆过程
   醇卤代写 R-OH + HX → R-X + H2O，消去写 R-X → 烯烃 + HX），
  NaOH/浓硫酸等仅作为条件关键词，不作为反应物。
- 不对称烯烃 HX/H2O 加成只出马氏规则主产物；卤代烃/醇消去按位点枚举全部
  不同烯烃；苯环取代结合定位效应只取有利位点（邻对位/间位）。

输入约定：reactants 为 Molecule 列表，顺序与规则的 input_index 对应，例如
酯化 [羧酸, 醇/苯酚]、加成类 [有机物, 试剂]、水解 [有机物, 水]、氧化 [有机物, O2]。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import organic_chemistry as oc


# ------- 常量与基础工具 -------

_HALOGENS: frozenset[str] = frozenset({"f", "cl", "br", "i"})

_HALOGEN_CN: dict[str, str] = {
    "f": "氟化氢",
    "cl": "氯化氢",
    "br": "溴化氢",
    "i": "碘化氢",
}

_H2_FORMULA: dict[str, int] = {"h": 2}
_O2_FORMULA: dict[str, int] = {"o": 2}
_H2O_FORMULA: dict[str, int] = {"h": 2, "o": 1}
_HNO3_FORMULA: dict[str, int] = {"h": 1, "n": 1, "o": 3}
_BR2_FORMULA: dict[str, int] = {"br": 2}


def _is_halogen(atom: oc.Atom) -> bool:
    """是否卤素原子。"""
    return atom.name in _HALOGENS


def _total_h(atom: oc.Atom) -> int:
    """总氢数 = 隐氢 + 显式 H 邻居数。"""
    explicit = sum(1 for bond in atom.bonds if bond.other(atom).name == "h")
    return atom.implicit_h + explicit


def _in_pi(atom: oc.Atom) -> bool:
    """原子是否属于某个 π 体系。"""
    return any(atom in pi.atoms for pi in atom.belong.pi_systems)


def _has_multiple_bond(atom: oc.Atom) -> bool:
    """原子是否带有显式多重键。"""
    return any(bond.order > 1 for bond in atom.bonds)


def _is_carbonyl_carbon(atom: oc.Atom) -> bool:
    """是否为羰基碳（直接双键连 O）。"""
    return any(bond.other(atom).name == "o" and bond.order == 2 for bond in atom.bonds)


def _has_single_o_neighbor(atom: oc.Atom) -> bool:
    """是否带单键 O 邻居（如羧基的 C-OH，用于排除羧酸/甲酸）。"""
    return any(bond.other(atom).name == "o" and bond.order == 1 for bond in atom.bonds)


def _hx_molecule(x_name: str) -> oc.Molecule:
    """构造显式 HX 副产物分子（如氯化氢）。"""
    molecule = oc.Molecule(name=_HALOGEN_CN.get(x_name, "卤化氢"))
    oc.add_bond(oc.Atom("h", molecule), oc.Atom(x_name, molecule))
    return molecule


def _nonzero_formula(formula: dict[str, int]) -> dict[str, int]:
    """去掉计数为 0 的条目（Molecule.formula 恒含 h:0，比较时忽略）。"""
    return {name: count for name, count in formula.items() if count != 0}


def _same_formula(actual: dict[str, int], expected: dict[str, int]) -> bool:
    return _nonzero_formula(actual) == _nonzero_formula(expected)


# ------- 反应物与规则的数据结构 -------


@dataclass
class ReactantSpec:
    """一条规则中某个反应物的规格。

    pattern 与 formula 二选一：pattern 用于子结构匹配（有机分子），formula
    用于整分子式匹配（小分子试剂）；count 为该反应物的化学计量数（引擎按
    计量数复制输入分子）；input_index 为输入 reactants 列表中的序号。
    """

    input_index: int
    count: int = 1
    pattern: oc.Pattern | None = None
    formula: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if (self.pattern is None) == (self.formula is None):
            raise ValueError("ReactantSpec 必须且只能指定 pattern 或 formula 之一")
        if self.count < 1:
            raise ValueError("ReactantSpec.count 必须 >= 1")
        if self.input_index < 0:
            raise ValueError("ReactantSpec.input_index 必须 >= 0")


class _SpecMatch:
    """一次“副本 × 位点”匹配；整分子式规格的副本整体即一个匹配。"""

    __slots__ = ("spec_index", "copy_index", "copy", "atom_map", "pi_map")

    def __init__(
        self,
        spec_index: int,
        copy_index: int,
        copy: oc.Molecule,
        atom_map: dict[oc.PatternAtom, oc.Atom],
        pi_map: dict[int, oc.PiSystem],
    ) -> None:
        self.spec_index: int = spec_index
        self.copy_index: int = copy_index
        self.copy: oc.Molecule = copy
        self.atom_map: dict[oc.PatternAtom, oc.Atom] = atom_map
        self.pi_map: dict[int, oc.PiSystem] = pi_map


class ReactionContext:
    """规则变换上下文：工作分子 + 全部副本匹配 + 副本→工作映射。"""

    def __init__(
        self,
        working: oc.Molecule,
        matches: Sequence[_SpecMatch],
        copy_map: dict[oc.Atom, oc.Atom],
        pi_merge: dict[oc.PiSystem, oc.PiSystem],
    ) -> None:
        self.working: oc.Molecule = working
        self.matches: tuple[_SpecMatch, ...] = tuple(matches)
        self.copy_map: dict[oc.Atom, oc.Atom] = copy_map
        self.pi_merge: dict[oc.PiSystem, oc.PiSystem] = pi_merge

    def atom(
        self,
        spec_index: int,
        pattern_atom: oc.PatternAtom,
        copy_index: int = 0,
    ) -> oc.Atom:
        """模式原子 → 合并工作分子中的原子。"""
        for match in self.matches:
            if match.spec_index == spec_index and match.copy_index == copy_index:
                return self.copy_map[match.atom_map[pattern_atom]]
        raise KeyError(f"规格 {spec_index} 副本 {copy_index} 的匹配不存在")

    def spec_atoms(self, spec_index: int) -> list[oc.Atom]:
        """某个规格全部副本的全部原子（整分子式规格用）。"""
        result: list[oc.Atom] = []
        for match in self.matches:
            if match.spec_index == spec_index:
                result.extend(self.copy_map[atom] for atom in match.copy.atoms)
        return result

    def ring_pi(self, spec_index: int = 0) -> oc.PiSystem | None:
        """该规格第一份副本中 pi_group=1 对应的合并 π 体系。"""
        for match in self.matches:
            if match.spec_index == spec_index:
                copy_pi = match.pi_map.get(1)
                if copy_pi is not None:
                    return self.pi_merge.get(copy_pi)
        return None


class ReactionRule:
    """一条反应规则：元数据 + 反应物规格 + 变换函数。"""

    def __init__(
        self,
        name: str,
        category: str,
        conditions: str,
        reactants: Sequence[ReactantSpec],
        apply: Callable[[ReactionContext], list[oc.Molecule]],
    ) -> None:
        self.name: str = name
        self.category: str = category
        self.conditions: str = conditions
        self.reactants: tuple[ReactantSpec, ...] = tuple(reactants)
        self._apply: Callable[[ReactionContext], list[oc.Molecule]] = apply
        if not self.reactants:
            raise ValueError("反应规则至少需要一个反应物规格")
        self.required_inputs: int = max(spec.input_index for spec in self.reactants) + 1

    def apply(self, ctx: ReactionContext) -> list[oc.Molecule]:
        """在工作分子上执行变换，返回额外显式构造的副产物。"""
        return list(self._apply(ctx))

    def __repr__(self) -> str:
        return f"<ReactionRule {self.name} [{self.category}]>"


@dataclass
class ReactionOutcome:
    """一次反应的全部产物（主产物 + 副产物，对应一次位点组合）。"""

    rule: ReactionRule
    products: tuple[oc.Molecule, ...]


# ------- 苯环定位工具 -------


def _ring_cycle(ring_atoms: Sequence[oc.Atom]) -> list[oc.Atom] | None:
    """把环原子按成键顺序排成一圈；无法成环时返回 None。"""
    ring_set = set(ring_atoms)
    if len(ring_set) < 3:
        return None
    start = ring_atoms[0]
    neighbors = [
        bond.other(start) for bond in start.bonds if bond.other(start) in ring_set
    ]
    if len(neighbors) != 2:
        return None
    cycle: list[oc.Atom] = [start]
    prev, current = start, neighbors[0]
    while current is not start:
        cycle.append(current)
        nxt: oc.Atom | None = None
        for bond in current.bonds:
            candidate = bond.other(current)
            if candidate in ring_set and candidate is not prev:
                nxt = candidate
                break
        if nxt is None:
            return None
        prev, current = current, nxt
        if len(cycle) > len(ring_set):
            return None
    return cycle if len(cycle) == len(ring_set) else None


def _directing_of(neighbor: oc.Atom) -> str:
    """按环外邻居原子粗略分类定位效应：ortho_para（邻对位）/ meta（间位）。"""
    if neighbor.name in _HALOGENS or neighbor.name == "o":
        return "ortho_para"
    if neighbor.name == "n":
        # 硝基 N（连 O）为间位定位基；氨基为邻对位定位基
        if any(bond.other(neighbor).name == "o" for bond in neighbor.bonds):
            return "meta"
        return "ortho_para"
    if neighbor.name == "c":
        # 羰基类（醛/酮/羧/酯）为间位定位基；烷基/苄基类为邻对位定位基
        return "meta" if _is_carbonyl_carbon(neighbor) else "ortho_para"
    return "ortho_para"


def _allowed_ring_positions(cycle: Sequence[oc.Atom]) -> set[int]:
    """返回苯环上允许亲电取代的位点下标（结合定位效应，且位点带 H）。"""
    ring_set = set(cycle)
    substituents: dict[int, oc.Atom] = {}
    for index, atom in enumerate(cycle):
        for bond in atom.bonds:
            neighbor = bond.other(atom)
            if neighbor in ring_set or neighbor.name == "h":
                continue
            substituents[index] = neighbor
            break
    if not substituents:
        return {index for index in range(len(cycle)) if _total_h(cycle[index]) >= 1}
    has_meta = any(
        _directing_of(neighbor) == "meta" for neighbor in substituents.values()
    )
    allowed: set[int] = set()
    for index in substituents:
        if has_meta:
            allowed |= {(index + 2) % len(cycle), (index - 2) % len(cycle)}
        else:
            allowed |= {
                (index + 1) % len(cycle),
                (index - 1) % len(cycle),
                (index + 3) % len(cycle),  # 对位 = 隔 3 个键
            }
    return {index for index in allowed if _total_h(cycle[index]) >= 1}


# ------- 连通分量拆分 -------


def _extract_component(source: oc.Molecule, atoms: Sequence[oc.Atom]) -> oc.Molecule:
    """从源分子提取一个连通分量（原子子集）到新分子。"""
    atom_set = set(atoms)
    result = oc.Molecule()
    mapping: dict[oc.Atom, oc.Atom] = {}
    for atom in atoms:
        mapping[atom] = (
            oc.ActiveH(result) if isinstance(atom, oc.ActiveH) else oc.Atom(atom.name, result)
        )
    for bond in source.bonds:
        atom1, atom2 = bond.atoms
        if atom1 in atom_set and atom2 in atom_set:
            oc.add_bond(mapping[atom1], mapping[atom2], order=bond.order)
    for pi in source.pi_systems:
        members = [mapping[atom] for atom in pi.atoms if atom in atom_set]
        if len(members) >= 2:
            oc.add_pi_system(members, dbe=pi.dbe)
    return result


def _split_components(molecule: oc.Molecule) -> list[oc.Molecule]:
    """按连通分量拆分分子，返回独立的新分子列表（保留键级与 π 体系）。"""
    seen: set[oc.Atom] = set()
    components: list[list[oc.Atom]] = []
    for atom in molecule.atoms:
        if atom in seen:
            continue
        component: list[oc.Atom] = []
        stack: list[oc.Atom] = [atom]
        seen.add(atom)
        while stack:
            current = stack.pop()
            component.append(current)
            for bond in current.bonds:
                neighbor = bond.other(current)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return [_extract_component(molecule, component) for component in components]


# ------- 引擎 -------


def _matches_for_spec(
    spec: ReactantSpec,
    spec_index: int,
    reactants: Sequence[oc.Molecule],
) -> list[list[_SpecMatch]]:
    """返回每个副本（按计量数复制）的匹配列表；外层列表对应每份副本。"""
    molecule = reactants[spec.input_index]
    groups: list[list[_SpecMatch]] = []
    for copy_index in range(spec.count):
        copy = oc.copy_molecule(molecule)
        if spec.pattern is not None:
            matches = oc.find_substructure_matches(spec.pattern, copy)
            groups.append(
                [
                    _SpecMatch(spec_index, copy_index, copy, match.atom_map, match.pi_map)
                    for match in matches
                ]
            )
        else:
            assert spec.formula is not None
            groups.append(
                [_SpecMatch(spec_index, copy_index, copy, {}, {})]
                if _same_formula(copy.formula, spec.formula)
                else []
            )
    return groups


def _cartesian(groups: Sequence[Sequence[_SpecMatch]]) -> list[tuple[_SpecMatch, ...]]:
    """各副本位点列表的笛卡尔积。"""
    combos: list[tuple[_SpecMatch, ...]] = [()]
    for group in groups:
        combos = [combo + (match,) for combo in combos for match in group]
    return combos


def _run_combo(
    rule: ReactionRule,
    combo: tuple[_SpecMatch, ...],
) -> list[oc.Molecule]:
    """执行一次位点组合：合并 → 变换 → 拆分 → 追加副产物。"""
    copy_map: dict[oc.Atom, oc.Atom] = {}
    pi_merge: dict[oc.PiSystem, oc.PiSystem] = {}
    working = oc.merge_molecules(
        [match.copy for match in combo],
        atom_map=copy_map,
        pi_map=pi_merge,
    )
    ctx = ReactionContext(working, combo, copy_map, pi_merge)
    extra = rule.apply(ctx)
    working.validate()
    return _split_components(working) + extra


_WATER_FORMULA: dict[str, int] = {"h": 2, "o": 1}


def _name_products(
    rule: ReactionRule,
    products: Sequence[oc.Molecule],
) -> tuple[oc.Molecule, ...]:
    """给产物命名：水副产物叫“水”，其余主产物按规则名编号。"""
    index = 1
    for product in products:
        if _same_formula(product.formula, _WATER_FORMULA):
            product.name = "水"
        elif product.name is None:
            product.name = f"{rule.name}产物{index}"
            index += 1
    return tuple(products)


def _outcomes_for_rule(
    rule: ReactionRule,
    reactants: Sequence[oc.Molecule],
) -> tuple[bool, list[ReactionOutcome]]:
    """返回 (是否有位点命中, 该规则的全部反应结果)。"""
    groups: list[list[_SpecMatch]] = []
    for spec_index, spec in enumerate(rule.reactants):
        groups.extend(_matches_for_spec(spec, spec_index, reactants))
    if not groups or not all(groups):
        return False, []
    outcomes: list[ReactionOutcome] = []
    for combo in _cartesian(groups):
        try:
            products = _run_combo(rule, combo)
        except ValueError:
            continue  # 位点不可行（价键/定位/试剂守卫），跳过该组合
        if not products:
            continue
        outcomes.append(ReactionOutcome(rule, _name_products(rule, products)))
    return True, outcomes


def _dedup(molecules: Sequence[oc.Molecule]) -> list[oc.Molecule]:
    """按结构严格判等去重，保持首次出现顺序。"""
    result: list[oc.Molecule] = []
    for molecule in molecules:
        if not any(molecule == other for other in result):
            result.append(molecule)
    return result


def _split_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[、,，;；/\s]+", text) if token]


def _check_reactants(reactants: Sequence[oc.Molecule]) -> list[oc.Molecule]:
    if not isinstance(reactants, (list, tuple)):
        raise ValueError("reactants 必须是 Molecule 列表或元组")
    result = list(reactants)
    if not result:
        raise ValueError("至少需要一个反应物分子")
    for molecule in result:
        if not isinstance(molecule, oc.Molecule):
            raise ValueError("反应物必须是 organic_chemistry.Molecule 实例")
    return result


# ------- 公共 API -------


def list_reactions() -> list[ReactionRule]:
    """返回全部反应规则。"""
    return list(RULES)


def find_reactions(
    reactants: Sequence[oc.Molecule],
    *,
    reaction: str | None = None,
    conditions: str | None = None,
    category: str | None = None,
) -> list[ReactionRule]:
    """按反应名/类别/条件过滤规则（不做分子匹配；无过滤时返回全部规则）。"""
    rules = list(RULES)
    if reaction is not None:
        key = reaction.strip()
        rules = [
            rule
            for rule in rules
            if key and (key == rule.name or key in rule.name or key == rule.category)
        ]
    if category is not None:
        key = category.strip()
        rules = [rule for rule in rules if key and rule.category == key]
    if conditions is not None:
        tokens = _split_tokens(conditions)
        rules = [rule for rule in rules if all(token in rule.conditions for token in tokens)]
    return rules


def reaction_outcomes(
    reactants: Sequence[oc.Molecule],
    *,
    reaction: str | None = None,
    conditions: str | None = None,
    category: str | None = None,
) -> list[ReactionOutcome]:
    """模拟反应，返回全部位点组合的反应结果（含规则与产物）。

    无符合条件的规则、反应物数量不符或反应物与所选反应完全不匹配时抛
    ValueError；有位点命中但全部位点组合不可行时返回空列表。
    """
    reactant_list = _check_reactants(reactants)
    rules = find_reactions(
        reactant_list,
        reaction=reaction,
        conditions=conditions,
        category=category,
    )
    if not rules:
        raise ValueError("未找到符合反应类型/条件的反应规则")
    count_ok = [rule for rule in rules if len(reactant_list) == rule.required_inputs]
    if not count_ok:
        needs = sorted({rule.required_inputs for rule in rules})
        raise ValueError(
            f"反应物数量不符：所选反应需要 {needs} 个反应物，实际 {len(reactant_list)} 个"
        )
    outcomes: list[ReactionOutcome] = []
    matched_any = False
    for rule in count_ok:
        matched, rule_outcomes = _outcomes_for_rule(rule, reactant_list)
        matched_any = matched_any or matched
        outcomes.extend(rule_outcomes)
    if not matched_any:
        names = "、".join(rule.name for rule in count_ok)
        raise ValueError(f"反应物分子与所选反应不匹配：{names}")
    return outcomes


def simulate_reaction(
    reactants: Sequence[oc.Molecule],
    *,
    reaction: str | None = None,
    conditions: str | None = None,
    category: str | None = None,
) -> list[oc.Molecule]:
    """模拟反应，返回去重后的全部生成物分子（含副产物）。"""
    outcomes = reaction_outcomes(
        reactants,
        reaction=reaction,
        conditions=conditions,
        category=category,
    )
    products: list[oc.Molecule] = []
    for outcome in outcomes:
        products.extend(outcome.products)
    return _dedup(products)


# ------- 反应规则目录 -------


def _build_rules() -> tuple[ReactionRule, ...]:
    rules: list[ReactionRule] = []

    # ---------- 加成 ----------

    def rule_alkene_h2() -> ReactionRule:
        pattern, c1, c2 = _pattern_alkene()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            a1 = ctx.atom(0, c1)
            a2 = ctx.atom(0, c2)
            oc.break_bond(a1, a2, order=1)  # C=C -> C-C
            for h in ctx.spec_atoms(1):
                oc.del_atom(h)
            return []

        return ReactionRule(
            "烯烃加氢", "加成", "催化剂/加热",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, formula=_H2_FORMULA)],
            apply,
        )

    def rule_alkene_x2() -> ReactionRule:
        pattern, c1, c2 = _pattern_alkene()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            a1 = ctx.atom(0, c1)
            a2 = ctx.atom(0, c2)
            x1, x2 = ctx.spec_atoms(1)
            if not (_is_halogen(x1) and _is_halogen(x2) and x1.name == x2.name):
                raise ValueError("X2 反应物必须是卤素单质")
            oc.break_bond(a1, a2, order=1)
            oc.break_bond(x1, x2)
            oc.add_bond(a1, x1)
            oc.add_bond(a2, x2)
            return []

        return ReactionRule(
            "烯烃与卤素加成", "加成", "",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, pattern=_pattern_x2()[0])],
            apply,
        )

    def rule_alkene_hx() -> ReactionRule:
        alkene, c1, c2 = _pattern_alkene()
        hx, h, x = _pattern_hx()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            a1 = ctx.atom(0, c1)
            a2 = ctx.atom(0, c2)
            h_atom = ctx.atom(1, h)
            x_atom = ctx.atom(1, x)
            if not _is_halogen(x_atom):
                raise ValueError("HX 反应物中的 X 必须是卤素")
            # 马氏规则：卤素加到氢较少的碳上
            if _total_h(a1) <= _total_h(a2):
                xc, hc = a1, a2
            else:
                xc, hc = a2, a1
            oc.break_bond(a1, a2, order=1)
            oc.break_bond(h_atom, x_atom)
            oc.add_bond(xc, x_atom)
            oc.del_atom(h_atom)
            return []

        return ReactionRule(
            "烯烃与HX加成", "加成", "",
            [ReactantSpec(0, pattern=alkene), ReactantSpec(1, pattern=hx)],
            apply,
        )

    def rule_alkene_h2o() -> ReactionRule:
        alkene, c1, c2 = _pattern_alkene()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            a1 = ctx.atom(0, c1)
            a2 = ctx.atom(0, c2)
            water_atoms = ctx.spec_atoms(1)
            o_atom = next(atom for atom in water_atoms if atom.name == "o")
            h_atoms = [atom for atom in water_atoms if atom.name == "h"]
            # 马氏规则：羟基加到氢较少的碳上
            if _total_h(a1) <= _total_h(a2):
                oh_c = a1
            else:
                oh_c = a2
            oc.break_bond(a1, a2, order=1)
            for h in h_atoms:
                oc.del_atom(h)
            oc.add_bond(oh_c, o_atom)
            return []

        return ReactionRule(
            "烯烃水化", "加成", "催化剂/加热",
            [ReactantSpec(0, pattern=alkene), ReactantSpec(1, formula=_H2O_FORMULA)],
            apply,
        )

    def rule_alkyne_h2(partial: bool) -> ReactionRule:
        pattern, c1, c2 = _pattern_alkyne()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            a1 = ctx.atom(0, c1)
            a2 = ctx.atom(0, c2)
            oc.break_bond(a1, a2, order=1 if partial else 2)  # C≡C -> C=C / C-C
            for h in ctx.spec_atoms(1):
                oc.del_atom(h)
            return []

        return ReactionRule(
            "炔烃部分加氢" if partial else "炔烃完全加氢",
            "加成",
            "催化剂/加热",
            [
                ReactantSpec(0, pattern=pattern),
                ReactantSpec(1, count=1 if partial else 2, formula=_H2_FORMULA),
            ],
            apply,
        )

    def rule_benzene_h2() -> ReactionRule:
        pattern, ring = _pattern_benzene()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            pi = ctx.ring_pi(0)
            if pi is None:
                raise ValueError("未找到苯环 π 体系")
            oc.remove_pi_system(pi)
            for h in ctx.spec_atoms(1):
                oc.del_atom(h)
            return []

        return ReactionRule(
            "苯加氢", "加成", "Ni/加热",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, count=3, formula=_H2_FORMULA)],
            apply,
        )

    # ---------- 取代 ----------

    def rule_alkane_halogenation() -> ReactionRule:
        pattern, c = _pattern_alkane_h()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            carbon = ctx.atom(0, c)
            x1, x2 = ctx.spec_atoms(1)
            if not (_is_halogen(x1) and _is_halogen(x2) and x1.name == x2.name):
                raise ValueError("X2 反应物必须是卤素单质")
            if _has_multiple_bond(carbon) or _in_pi(carbon):
                raise ValueError("烷烃卤代只作用于饱和碳")
            oc.break_bond(x1, x2)
            oc.add_bond(carbon, x1)
            oc.del_atom(x2)
            return [_hx_molecule(x1.name)]

        return ReactionRule(
            "烷烃卤代", "取代", "光照",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, pattern=_pattern_x2()[0])],
            apply,
        )

    def rule_benzene_halogenation() -> ReactionRule:
        pattern, c = _pattern_ring_c()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            carbon = ctx.atom(0, c)
            x1, x2 = ctx.spec_atoms(1)
            if not (_is_halogen(x1) and _is_halogen(x2) and x1.name == x2.name):
                raise ValueError("X2 反应物必须是卤素单质")
            pi = ctx.ring_pi(0)
            if pi is None:
                raise ValueError("未找到苯环 π 体系")
            cycle = _ring_cycle(pi.atoms)
            if cycle is None or cycle.index(carbon) not in _allowed_ring_positions(cycle):
                raise ValueError("该苯环位点不适合发生卤代")
            oc.break_bond(x1, x2)
            oc.add_bond(carbon, x1)
            oc.del_atom(x2)
            return [_hx_molecule(x1.name)]

        return ReactionRule(
            "苯卤代", "取代", "FeX3/加热",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, pattern=_pattern_x2()[0])],
            apply,
        )

    def rule_benzene_nitration() -> ReactionRule:
        pattern, c = _pattern_ring_c()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            carbon = ctx.atom(0, c)
            pi = ctx.ring_pi(0)
            if pi is None:
                raise ValueError("未找到苯环 π 体系")
            cycle = _ring_cycle(pi.atoms)
            if cycle is None or cycle.index(carbon) not in _allowed_ring_positions(cycle):
                raise ValueError("该苯环位点不适合发生硝化")
            n_atom = next(atom for atom in ctx.spec_atoms(1) if atom.name == "n")
            # HNO3 中不在硝基 π 体系里的 O 即羟基氧（-OH）
            o_h: oc.Atom | None = None
            for bond in n_atom.bonds:
                candidate = bond.other(n_atom)
                if candidate.name == "o" and not _in_pi(candidate):
                    o_h = candidate
                    break
            if o_h is None:
                raise ValueError("HNO3 结构异常：找不到羟基氧")
            oc.break_bond(n_atom, o_h)
            oc.add_bond(carbon, n_atom)
            return []

        return ReactionRule(
            "苯硝化", "取代", "浓硫酸/加热",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, formula=_HNO3_FORMULA)],
            apply,
        )

    def rule_alcohol_halogenation() -> ReactionRule:
        alcohol, c, o = _pattern_alcohol()
        hx, h, x = _pattern_hx()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            carbon = ctx.atom(0, c)
            o_atom = ctx.atom(0, o)
            h_atom = ctx.atom(1, h)
            x_atom = ctx.atom(1, x)
            if not _is_halogen(x_atom):
                raise ValueError("醇卤代要求 HX 中的 X 为卤素")
            if _is_carbonyl_carbon(carbon):
                raise ValueError("醇卤代不适用于羧酸羟基")
            oc.break_bond(carbon, o_atom)  # 脱去羟基
            oc.break_bond(h_atom, x_atom)
            oc.add_bond(carbon, x_atom)
            # HX 的 H 与脱去的羟基组成水，由隐氢自动平衡，故删除该游离氢
            oc.del_atom(h_atom)
            return []

        return ReactionRule(
            "醇卤代", "取代", "加热",
            [ReactantSpec(0, pattern=alcohol), ReactantSpec(1, pattern=hx)],
            apply,
        )

    def rule_alcohol_etherification() -> ReactionRule:
        pattern, c, o = _pattern_alcohol()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            c0 = ctx.atom(0, c, 0)
            o0 = ctx.atom(0, o, 0)
            c1 = ctx.atom(0, c, 1)
            o1 = ctx.atom(0, o, 1)
            for carbon in (c0, c1):
                if _is_carbonyl_carbon(carbon):
                    raise ValueError("分子间脱水不适用于羧酸羟基")
            oc.break_bond(c1, o1)
            oc.add_bond(o0, c1)  # R1-O-R2
            return []

        return ReactionRule(
            "醇分子间脱水", "取代", "浓硫酸/140℃/加热",
            [ReactantSpec(0, count=2, pattern=pattern)],
            apply,
        )

    def rule_phenol_bromine() -> ReactionRule:
        pattern, ring, o = _pattern_phenol()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            ring_atoms = [ctx.atom(0, pa) for pa in ring]
            o_atom = ctx.atom(0, o)
            cycle = _ring_cycle(ring_atoms)
            if cycle is None:
                raise ValueError("苯酚环无法成环")
            # 找到连羟基的环碳并旋转使其位于下标 0，三溴位点 = 2,4,6
            ring_set = set(cycle)
            o_carbon = next(
                (bond.other(o_atom) for bond in o_atom.bonds if bond.other(o_atom) in ring_set),
                None,
            )
            if o_carbon is None:
                raise ValueError("苯酚结构异常：羟基未连在苯环上")
            start = cycle.index(o_carbon)
            rotated = cycle[start:] + cycle[:start]
            targets = [rotated[1], rotated[3], rotated[5]]
            if any(_total_h(target) < 1 for target in targets):
                raise ValueError("邻对位无足够氢，不能生成三溴苯酚")
            br_atoms = ctx.spec_atoms(1)
            extra: list[oc.Molecule] = []
            for index, target in enumerate(targets):
                b1, b2 = br_atoms[2 * index], br_atoms[2 * index + 1]
                oc.break_bond(b1, b2)
                oc.add_bond(target, b1)
                oc.del_atom(b2)
                extra.append(_hx_molecule(b1.name))
            return extra

        return ReactionRule(
            "苯酚与溴水", "取代", "溴水",
            [
                ReactantSpec(0, pattern=pattern),
                ReactantSpec(1, count=3, formula=_BR2_FORMULA),
            ],
            apply,
        )

    # ---------- 消去 ----------

    def rule_haloalkane_elimination() -> ReactionRule:
        pattern, c1, c2, x = _pattern_elimination()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            a1 = ctx.atom(0, c1)
            a2 = ctx.atom(0, c2)
            x_atom = ctx.atom(0, x)
            if not _is_halogen(x_atom):
                raise ValueError("卤代烃消去要求 C-X 中的 X 为卤素")
            if _in_pi(a1) or _in_pi(a2):
                raise ValueError("卤代烃消去只作用于饱和碳")
            oc.break_bond(a2, x_atom)
            oc.add_bond(a1, a2)  # 升为双键，β-H 由隐氢自动扣除
            oc.del_atom(x_atom)
            return [_hx_molecule(x_atom.name)]

        return ReactionRule(
            "卤代烃消去", "消去", "NaOH醇溶液/加热",
            [ReactantSpec(0, pattern=pattern)],
            apply,
        )

    def rule_alcohol_dehydration() -> ReactionRule:
        pattern, c1, c2, o = _pattern_dehydration()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            a1 = ctx.atom(0, c1)
            a2 = ctx.atom(0, c2)
            o_atom = ctx.atom(0, o)
            if _is_carbonyl_carbon(a2):
                raise ValueError("醇分子内脱水不适用于羧酸羟基")
            oc.break_bond(a2, o_atom)
            oc.add_bond(a1, a2)  # 升为双键
            return []

        return ReactionRule(
            "醇分子内脱水", "消去", "浓硫酸/170℃/加热",
            [ReactantSpec(0, pattern=pattern)],
            apply,
        )

    # ---------- 氧化 ----------

    def rule_alcohol_oxidation() -> ReactionRule:
        pattern, c, o = _pattern_alcohol()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            for match in ctx.matches:
                if match.spec_index != 0:
                    continue
                carbon = ctx.copy_map[match.atom_map[c]]
                o_atom = ctx.copy_map[match.atom_map[o]]
                if _is_carbonyl_carbon(carbon):
                    raise ValueError("醇催化氧化不适用于羧酸羟基")
                oc.add_bond(carbon, o_atom)  # C-O -> C=O
            o1, o2 = ctx.spec_atoms(1)
            oc.break_bond(o1, o2)  # O=O 断裂，两个 O 各自成为水
            return []

        return ReactionRule(
            "醇催化氧化", "氧化", "Cu或Ag/加热",
            [
                ReactantSpec(0, count=2, pattern=pattern),
                ReactantSpec(1, formula=_O2_FORMULA),
            ],
            apply,
        )

    def rule_aldehyde_oxidation() -> ReactionRule:
        pattern, c, o = _pattern_aldehyde()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            o1, o2 = ctx.spec_atoms(1)
            oc.break_bond(o1, o2)
            aldehydes = [match for match in ctx.matches if match.spec_index == 0]
            for match, o_atom in zip(aldehydes, (o1, o2)):
                carbon = ctx.copy_map[match.atom_map[c]]
                if _has_single_o_neighbor(carbon):
                    raise ValueError("醛催化氧化不适用于羧酸（含甲酸）")
                oc.add_bond(carbon, o_atom)  # C-H + O -> C-OH
            return []

        return ReactionRule(
            "醛催化氧化", "氧化", "催化剂/加热",
            [
                ReactantSpec(0, count=2, pattern=pattern),
                ReactantSpec(1, formula=_O2_FORMULA),
            ],
            apply,
        )

    # ---------- 还原 ----------

    def rule_aldehyde_hydrogenation() -> ReactionRule:
        pattern, c, o = _pattern_aldehyde()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            carbon = ctx.atom(0, c)
            o_atom = ctx.atom(0, o)
            if _has_single_o_neighbor(carbon):
                raise ValueError("醛加氢还原不适用于羧酸（含甲酸）")
            oc.break_bond(carbon, o_atom, order=1)  # C=O -> C-O
            for h in ctx.spec_atoms(1):
                oc.del_atom(h)
            return []

        return ReactionRule(
            "醛加氢还原", "还原", "催化剂/加热",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, formula=_H2_FORMULA)],
            apply,
        )

    def rule_nitro_reduction() -> ReactionRule:
        pattern, n, o1, o2 = _pattern_nitro()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            pi = ctx.ring_pi(0)
            if pi is None:
                raise ValueError("未找到硝基 π 体系")
            n_atom = ctx.atom(0, n)
            oa = ctx.atom(0, o1)
            ob = ctx.atom(0, o2)
            oc.remove_pi_system(pi)
            oc.break_bond(n_atom, oa)
            oc.break_bond(n_atom, ob)
            for h in ctx.spec_atoms(1):
                oc.del_atom(h)
            return []

        return ReactionRule(
            "硝基还原", "还原", "催化剂/加热",
            [
                ReactantSpec(0, pattern=pattern),
                ReactantSpec(1, count=3, formula=_H2_FORMULA),
            ],
            apply,
        )

    # ---------- 水解 ----------

    def rule_haloalkane_hydrolysis() -> ReactionRule:
        pattern, c, x = _pattern_haloalkane()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            carbon = ctx.atom(0, c)
            x_atom = ctx.atom(0, x)
            if not _is_halogen(x_atom):
                raise ValueError("卤代烃水解要求 C-X 中的 X 为卤素")
            if _in_pi(carbon) or _has_multiple_bond(carbon):
                raise ValueError("卤代烃水解只作用于饱和碳上的卤素")
            water_atoms = ctx.spec_atoms(1)
            o_atom = next(atom for atom in water_atoms if atom.name == "o")
            h_atoms = [atom for atom in water_atoms if atom.name == "h"]
            oc.break_bond(carbon, x_atom)
            oc.del_atom(x_atom)
            for h in h_atoms:
                oc.del_atom(h)
            oc.add_bond(carbon, o_atom)
            return [_hx_molecule(x_atom.name)]

        return ReactionRule(
            "卤代烃水解", "水解", "NaOH水溶液/加热",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, formula=_H2O_FORMULA)],
            apply,
        )

    def rule_ester_hydrolysis() -> ReactionRule:
        pattern, c, o_d, o_b, c_r = _pattern_ester()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            carbonyl = ctx.atom(0, c)
            bridge = ctx.atom(0, o_b)
            water_atoms = ctx.spec_atoms(1)
            o_atom = next(atom for atom in water_atoms if atom.name == "o")
            h_atoms = [atom for atom in water_atoms if atom.name == "h"]
            oc.break_bond(carbonyl, bridge)  # 酯基断键 -> 羧酸 + 醇
            for h in h_atoms:
                oc.del_atom(h)
            oc.add_bond(carbonyl, o_atom)
            return []

        return ReactionRule(
            "酯水解", "水解", "酸或碱催化/加热",
            [ReactantSpec(0, pattern=pattern), ReactantSpec(1, formula=_H2O_FORMULA)],
            apply,
        )

    # ---------- 酯化 ----------

    def rule_esterification() -> ReactionRule:
        acid_p, acid_c, acid_oh = _pattern_carboxyl()
        alc_p, alc_c, alc_o = _pattern_alcohol()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            acid_carbon = ctx.atom(0, acid_c)
            acid_o = ctx.atom(0, acid_oh)
            alc_carbon = ctx.atom(1, alc_c)
            alc_o_atom = ctx.atom(1, alc_o)
            if _is_carbonyl_carbon(alc_carbon):
                raise ValueError("酯化反应要求第二个反应物为醇（不能是羧酸）")
            oc.break_bond(acid_carbon, acid_o)  # 羧基脱 OH
            oc.add_bond(acid_carbon, alc_o_atom)  # 成酯键
            return []

        return ReactionRule(
            "酯化反应", "酯化", "浓硫酸/加热",
            [
                ReactantSpec(0, pattern=acid_p),
                ReactantSpec(1, pattern=alc_p),
            ],
            apply,
        )

    def rule_phenol_esterification() -> ReactionRule:
        acid_p, acid_c, acid_oh = _pattern_carboxyl()
        phenol_p, _ring, phenol_o = _pattern_phenol()

        def apply(ctx: ReactionContext) -> list[oc.Molecule]:
            acid_carbon = ctx.atom(0, acid_c)
            acid_o = ctx.atom(0, acid_oh)
            phenol_oxygen = ctx.atom(1, phenol_o)
            oc.break_bond(acid_carbon, acid_o)  # 羧基脱 OH
            oc.add_bond(acid_carbon, phenol_oxygen)  # 成酯键
            return []

        return ReactionRule(
            "苯酚酯化", "酯化", "浓硫酸/加热",
            [
                ReactantSpec(0, pattern=acid_p),
                ReactantSpec(1, pattern=phenol_p),
            ],
            apply,
        )

    # ---------- 目录 ----------

    rules.append(rule_alkene_h2())
    rules.append(rule_alkene_x2())
    rules.append(rule_alkene_hx())
    rules.append(rule_alkene_h2o())
    rules.append(rule_alkyne_h2(partial=True))
    rules.append(rule_alkyne_h2(partial=False))
    rules.append(rule_benzene_h2())
    rules.append(rule_alkane_halogenation())
    rules.append(rule_benzene_halogenation())
    rules.append(rule_benzene_nitration())
    rules.append(rule_alcohol_halogenation())
    rules.append(rule_alcohol_etherification())
    rules.append(rule_phenol_bromine())
    rules.append(rule_haloalkane_elimination())
    rules.append(rule_alcohol_dehydration())
    rules.append(rule_alcohol_oxidation())
    rules.append(rule_aldehyde_oxidation())
    rules.append(rule_aldehyde_hydrogenation())
    rules.append(rule_nitro_reduction())
    rules.append(rule_haloalkane_hydrolysis())
    rules.append(rule_ester_hydrolysis())
    rules.append(rule_esterification())
    rules.append(rule_phenol_esterification())
    return tuple(rules)


# ------- 模式构建 -------


def _pattern_alkene() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    c1 = p.add_atom("c")
    c2 = p.add_atom("c")
    p.add_bond(c1, c2, order=2)
    return p, c1, c2


def _pattern_alkyne() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    c1 = p.add_atom("c")
    c2 = p.add_atom("c")
    p.add_bond(c1, c2, order=3)
    return p, c1, c2


def _pattern_hx() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    h = p.add_atom("h")
    x = p.add_atom()
    p.add_bond(h, x)
    return p, h, x


def _pattern_x2() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    a = p.add_atom()
    b = p.add_atom()
    p.add_bond(a, b)
    return p, a, b


def _pattern_benzene() -> tuple[oc.Pattern, tuple[oc.PatternAtom, ...]]:
    p = oc.Pattern()
    ring = tuple(p.add_atom("c", pi=True, pi_group=1) for _ in range(6))
    for i in range(6):
        p.add_bond(ring[i], ring[(i + 1) % 6])
    return p, ring


def _pattern_ring_c() -> tuple[oc.Pattern, oc.PatternAtom]:
    """苯环上带氢的可取代碳（用于卤代/硝化，按位点枚举）。"""
    p = oc.Pattern()
    c = p.add_atom("c", h_min=1, pi=True, pi_group=1)
    return p, c


def _pattern_alkane_h() -> tuple[oc.Pattern, oc.PatternAtom]:
    """带氢的碳（卤代位点；apply 守卫要求饱和）。"""
    p = oc.Pattern()
    c = p.add_atom("c", h_min=1, pi=False)
    return p, c


def _pattern_alcohol() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    c = p.add_atom("c", h_min=1, pi=False)
    o = p.add_atom("o", h_min=1, pi=False)
    p.add_bond(c, o)
    return p, c, o


def _pattern_aldehyde() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    c = p.add_atom("c", h_min=1)
    o = p.add_atom("o")
    p.add_bond(c, o, order=2)
    return p, c, o


def _pattern_carboxyl() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    c = p.add_atom("c")
    o_d = p.add_atom("o")
    o_h = p.add_atom("o", h_min=1)
    p.add_bond(c, o_d, order=2)
    p.add_bond(c, o_h)
    return p, c, o_h


def _pattern_ester() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    c = p.add_atom("c")
    o_d = p.add_atom("o")
    o_b = p.add_atom("o")
    c_r = p.add_atom("c")
    p.add_bond(c, o_d, order=2)
    p.add_bond(c, o_b)
    p.add_bond(o_b, c_r)
    return p, c, o_d, o_b, c_r


def _pattern_nitro() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    n = p.add_atom("n", pi=True, pi_group=1)
    o1 = p.add_atom("o", pi=True, pi_group=1)
    o2 = p.add_atom("o", pi=True, pi_group=1)
    p.add_bond(n, o1)
    p.add_bond(n, o2)
    return p, n, o1, o2


def _pattern_haloalkane() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom]:
    p = oc.Pattern()
    c = p.add_atom("c", pi=False)
    x = p.add_atom()
    p.add_bond(c, x)
    return p, c, x


def _pattern_elimination() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom, oc.PatternAtom]:
    """C1-C2-X，其中 C1 带 β-H（消去位点）。"""
    p = oc.Pattern()
    c1 = p.add_atom("c", h_min=1, pi=False)
    c2 = p.add_atom("c", pi=False)
    x = p.add_atom()
    p.add_bond(c1, c2)
    p.add_bond(c2, x)
    return p, c1, c2, x


def _pattern_dehydration() -> tuple[oc.Pattern, oc.PatternAtom, oc.PatternAtom, oc.PatternAtom]:
    """C1-C2-OH，其中 C1 带 β-H（分子内脱水位点）。"""
    p = oc.Pattern()
    c1 = p.add_atom("c", h_min=1, pi=False)
    c2 = p.add_atom("c", pi=False)
    o = p.add_atom("o", h_min=1, pi=False)
    p.add_bond(c1, c2)
    p.add_bond(c2, o)
    return p, c1, c2, o


def _pattern_phenol() -> tuple[oc.Pattern, tuple[oc.PatternAtom, ...], oc.PatternAtom]:
    p = oc.Pattern()
    ring = tuple(p.add_atom("c", pi=True, pi_group=1) for _ in range(6))
    for i in range(6):
        p.add_bond(ring[i], ring[(i + 1) % 6])
    o = p.add_atom("o", h_min=1, pi=False)
    p.add_bond(ring[0], o)
    return p, ring, o


RULES: tuple[ReactionRule, ...] = _build_rules()


if __name__ == "__main__":
    print(f"已加载 {len(RULES)} 条反应规则")
    for rule in RULES:
        print(f"  [{rule.category}] {rule.name}（{rule.conditions}）")
