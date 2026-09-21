"""有机合成流程设计模块（无第三方依赖）。

输入起始反应物（含碳有机分子）、目标产物与可选的反应条件/反应名/类别，
输出不超过 max_steps（默认 6）步的合成路线，每步均为可被反应引擎真实模拟的反应。

搜索方式：双向 BFS。
- 正向侧：从起始反应物出发，对每条允许规则枚举“从当前状态选取有机反应物 +
  自动补全小分子试剂”的输入组合，调用 oc_reactions.reaction_outcomes 模拟反应；
- 逆向侧：内置反合成模板对目标分子逐级拆解，每个模板生成的候选反应物必须
  用对应正向规则跑一遍并做结构判等验证，验证通过才进入逆向状态；
- 汇合条件：正向状态 ⊇ 逆向所需分子集合，路线 = 正向路径 + 逆向路径反向。

状态约定（结构层面，不计计量数）：
- 状态 = 含碳有机结构的集合（去重）；副产有机物与试剂视为可丢弃或重复使用；
- 小分子试剂 H2 / O2 / H2O / HNO3 / Cl2 / Br2 / HCl / HBr 由内置目录自动补全，
  不计入步数、不需用户输入、不进入状态。

局限：束宽搜索可能漏解（找到的路线保证可真实模拟）；不严格平衡计量数与收率，
与 oc_reactions 的“等价表达”口径一致。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Callable

import oc_reactions as rx
import organic_chemistry as oc


# ------- 常量 -------

_HALOGENS: frozenset[str] = frozenset({"f", "cl", "br", "i"})
_ELEMENTS: tuple[str, ...] = ("c", "h", "n", "o", "f", "cl", "br", "i")

DEFAULT_MAX_STEPS: int = 6
DEFAULT_BEAM_WIDTH: int = 300
DEFAULT_MAX_ROUTES: int = 10
DEFAULT_NODE_BUDGET: int = 50_000

_Fp = tuple[str, ...]
_State = frozenset[_Fp]


# ------- 小分子试剂目录 -------


def _build_free_reagents() -> dict[str, oc.Molecule]:
    """构建自动补全的小分子试剂目录（按键名索引）。"""
    h2 = oc.Molecule(name="氢气")
    oc.add_bond(oc.Atom("h", h2), oc.Atom("h", h2))

    o2 = oc.Molecule(name="氧气")
    oc.add_bond(oc.Atom("o", o2), oc.Atom("o", o2), order=2)

    h2o = oc.Molecule(name="水")
    o = oc.Atom("o", h2o)
    oc.add_bond(o, oc.Atom("h", h2o))
    oc.add_bond(o, oc.Atom("h", h2o))

    hno3 = oc.Molecule(name="硝酸")
    n = oc.Atom("n", hno3)
    hno3_o1 = oc.Atom("o", hno3)
    hno3_o2 = oc.Atom("o", hno3)
    hno3_o3 = oc.Atom("o", hno3)
    oc.add_bond(n, hno3_o1)
    oc.add_bond(n, hno3_o2)
    oc.add_bond(n, hno3_o3)
    oc.add_bond(hno3_o3, oc.Atom("h", hno3))
    oc.add_pi_system([n, hno3_o1, hno3_o2])

    cl2 = oc.Molecule(name="氯气")
    oc.add_bond(oc.Atom("cl", cl2), oc.Atom("cl", cl2))

    br2 = oc.Molecule(name="溴")
    oc.add_bond(oc.Atom("br", br2), oc.Atom("br", br2))

    hcl = oc.Molecule(name="氯化氢")
    oc.add_bond(oc.Atom("h", hcl), oc.Atom("cl", hcl))

    hbr = oc.Molecule(name="溴化氢")
    oc.add_bond(oc.Atom("h", hbr), oc.Atom("br", hbr))

    return {
        "H2": h2,
        "O2": o2,
        "H2O": h2o,
        "HNO3": hno3,
        "Cl2": cl2,
        "Br2": br2,
        "HCl": hcl,
        "HBr": hbr,
    }


FREE_REAGENTS: dict[str, oc.Molecule] = _build_free_reagents()


# 每条规则各输入位的小分子试剂候选键名；空元组表示该位为有机反应物，
# 从当前状态中选取。长度必须等于规则 required_inputs。
_REAGENT_INPUTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "烯烃加氢": ((), ("H2",)),
    "烯烃与卤素加成": ((), ("Cl2", "Br2")),
    "烯烃与HX加成": ((), ("HCl", "HBr")),
    "烯烃水化": ((), ("H2O",)),
    "炔烃部分加氢": ((), ("H2",)),
    "炔烃完全加氢": ((), ("H2",)),
    "苯加氢": ((), ("H2",)),
    "烷烃卤代": ((), ("Cl2", "Br2")),
    "苯卤代": ((), ("Cl2", "Br2")),
    "苯硝化": ((), ("HNO3",)),
    "醇卤代": ((), ("HCl", "HBr")),
    "醇分子间脱水": ((),),
    "苯酚与溴水": ((), ("Br2",)),
    "卤代烃消去": ((),),
    "醇分子内脱水": ((),),
    "醇催化氧化": ((), ("O2",)),
    "醛催化氧化": ((), ("O2",)),
    "醛加氢还原": ((), ("H2",)),
    "硝基还原": ((), ("H2",)),
    "卤代烃水解": ((), ("H2O",)),
    "酯水解": ((), ("H2O",)),
    "酯化反应": ((), ()),
}


# ------- 基础工具 -------


def _has_carbon(molecule: oc.Molecule) -> bool:
    return molecule.formula.get("c", 0) > 0


def _fp(molecule: oc.Molecule) -> _Fp:
    return tuple(molecule.feature)


def _resolve_named(
    molecule: oc.Molecule,
    registry: dict[_Fp, oc.Molecule],
) -> oc.Molecule:
    """同结构分子优先取仓库中已登记的实例，以保留用户命名。

    反合成模板拆出的片段本身没有名称；若该结构已在仓库中（起始反应物、
    目标产物或此前某步的产物），改用登记实例，路线步骤里显示的就是原名。
    """
    return registry.get(_fp(molecule), molecule)


def _total_h(atom: oc.Atom) -> int:
    explicit = sum(1 for bond in atom.bonds if bond.other(atom).name == "h")
    return atom.implicit_h + explicit


def _in_pi(atom: oc.Atom) -> bool:
    return any(atom in pi.atoms for pi in atom.belong.pi_systems)


def _neighbors(atom: oc.Atom) -> list[oc.Atom]:
    return [bond.other(atom) for bond in atom.bonds]


def _is_halogen(atom: oc.Atom) -> bool:
    return atom.name in _HALOGENS


def _is_carbonyl_carbon(atom: oc.Atom) -> bool:
    return any(
        bond.other(atom).name == "o" and bond.order == 2 for bond in atom.bonds
    )


def _has_single_o_neighbor(atom: oc.Atom) -> bool:
    return any(
        bond.other(atom).name == "o" and bond.order == 1 for bond in atom.bonds
    )


def _has_multiple_bond(atom: oc.Atom) -> bool:
    return any(bond.order > 1 for bond in atom.bonds)


def _single_halogen_neighbor(atom: oc.Atom) -> oc.Atom | None:
    """原子的单键卤素邻居；恰好一个时返回，否则 None。"""
    halogens = [
        bond.other(atom)
        for bond in atom.bonds
        if bond.order == 1 and _is_halogen(bond.other(atom))
    ]
    return halogens[0] if len(halogens) == 1 else None


def _single_o_neighbor(atom: oc.Atom) -> oc.Atom | None:
    """原子的单键氧邻居；恰好一个时返回，否则 None。"""
    oxygens = [
        bond.other(atom)
        for bond in atom.bonds
        if bond.order == 1 and bond.other(atom).name == "o"
    ]
    return oxygens[0] if len(oxygens) == 1 else None


def _is_nitro_pi(atoms: Sequence[oc.Atom]) -> bool:
    """硝基型 π 体系：3 个成员且恰好 1 个 n、2 个 o。"""
    if len(atoms) != 3:
        return False
    counts: dict[str, int] = {}
    for atom in atoms:
        counts[atom.name] = counts.get(atom.name, 0) + 1
    return counts.get("n") == 1 and counts.get("o") == 2


def _element_counts(molecules: Sequence[oc.Molecule]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for molecule in molecules:
        for element, count in molecule.formula.items():
            if count:
                counts[element] = counts.get(element, 0) + count
    return counts


def _distance(
    state: _State,
    registry: dict[_Fp, oc.Molecule],
    reference: Sequence[oc.Molecule],
) -> int:
    """状态与参考分子集合的元素组成距离（用于束宽排序启发）。"""
    target_counts = _element_counts(reference)
    total = 0
    for key in state:
        formula = registry[key].formula
        for element in _ELEMENTS:
            total += abs(formula.get(element, 0) - target_counts.get(element, 0))
    return total


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


def _organic_components(molecule: oc.Molecule) -> list[oc.Molecule]:
    return [component for component in _split_components(molecule) if _has_carbon(component)]


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


def _find_carbon_cycle(molecule: oc.Molecule, length: int) -> list[oc.Atom] | None:
    """在分子中找一条长度为 length 的简单碳环（仅经过单键）。"""
    for start in molecule.atoms:
        if start.name != "c":
            continue
        path: list[oc.Atom] = [start]
        visited: set[oc.Atom] = {start}

        def dfs(node: oc.Atom) -> bool:
            if len(path) == length:
                return any(
                    bond.other(node) is start and bond.order == 1
                    for bond in node.bonds
                )
            for bond in node.bonds:
                if bond.order != 1:
                    continue
                nxt = bond.other(node)
                if nxt.name != "c" or nxt in visited:
                    continue
                path.append(nxt)
                visited.add(nxt)
                if dfs(nxt):
                    return True
                visited.remove(nxt)
                path.pop()
            return False

        if dfs(start):
            return list(path)
    return None


def _copy_with_map(molecule: oc.Molecule) -> tuple[oc.Molecule, dict[oc.Atom, oc.Atom]]:
    atom_map: dict[oc.Atom, oc.Atom] = {}
    clone = oc.copy_molecule(molecule, atom_map=atom_map)
    return clone, atom_map


# ------- 公共数据结构 -------


@dataclass(frozen=True)
class SynthesisStep:
    """合成路线中的一步反应。

    inputs 为传给反应引擎的完整输入（含自动补全的小分子试剂）；
    outputs 为引擎产物（含副产物）。
    """

    step_no: int
    rule_name: str
    category: str
    conditions: str
    inputs: tuple[oc.Molecule, ...]
    outputs: tuple[oc.Molecule, ...]


@dataclass(frozen=True)
class SynthesisRoute:
    """一条合成路线：从起始反应物到目标产物的有序步骤。"""

    target: oc.Molecule
    steps: tuple[SynthesisStep, ...]

    @property
    def step_count(self) -> int:
        return len(self.steps)


# ------- 搜索内部结构 -------


@dataclass
class _ForwardStep:
    """正向搜索中的一步：规则 + 实际输入 + 一次命中结果。"""

    rule: rx.ReactionRule
    inputs: tuple[oc.Molecule, ...]
    outcome: rx.ReactionOutcome


@dataclass
class _BackwardStep:
    """逆向搜索中的一步：验证用正向规则 + 实际输入 + 一次命中结果。"""

    rule: rx.ReactionRule
    inputs: tuple[oc.Molecule, ...]
    outcome: rx.ReactionOutcome


class _Budget:
    """搜索节点预算（两个方向共享）。"""

    def __init__(self, limit: int) -> None:
        self.limit: int = limit
        self.spent: int = 0

    def take(self) -> bool:
        if self.spent >= self.limit:
            return False
        self.spent += 1
        return True


# ------- 正向搜索 -------


def _forward_expand(
    state: _State,
    rules: Sequence[rx.ReactionRule],
    registry: dict[_Fp, oc.Molecule],
    visited: set[_State],
) -> dict[_State, _ForwardStep]:
    """对状态做一步正向展开：返回 新状态 -> 产生它的步骤。"""
    transitions: dict[_State, _ForwardStep] = {}
    state_mols = [registry[key] for key in sorted(state)]
    for rule in rules:
        reagent_groups = _REAGENT_INPUTS[rule.name]
        organic_indices = [index for index, keys in enumerate(reagent_groups) if not keys]
        if not organic_indices:
            continue
        choice_lists: list[tuple[str | None, ...]] = [
            tuple(keys) if keys else (None,) for keys in reagent_groups
        ]
        for combo in product(state_mols, repeat=len(organic_indices)):
            for choices in product(*choice_lists):
                inputs: list[oc.Molecule] = []
                combo_index = 0
                for index, keys in enumerate(reagent_groups):
                    if keys:
                        inputs.append(FREE_REAGENTS[choices[index]])
                    else:
                        inputs.append(combo[combo_index])
                        combo_index += 1
                try:
                    outcomes = rx.reaction_outcomes(inputs, reaction=rule.name)
                except ValueError:
                    continue
                for outcome in outcomes:
                    new_products: set[_Fp] = set()
                    for product_mol in outcome.products:
                        if not _has_carbon(product_mol):
                            continue
                        fp = _fp(product_mol)
                        registry.setdefault(fp, product_mol)
                        new_products.add(fp)
                    if not new_products:
                        continue
                    new_state = state | frozenset(new_products)
                    if new_state == state or new_state in visited:
                        continue
                    if new_state not in transitions:
                        transitions[new_state] = _ForwardStep(rule, tuple(inputs), outcome)
    return transitions


def _forward_layers(
    start_state: _State,
    target: oc.Molecule,
    registry: dict[_Fp, oc.Molecule],
    rules: Sequence[rx.ReactionRule],
    max_depth: int,
    beam_width: int,
    budget: _Budget,
) -> tuple[dict[_State, tuple[_State, _ForwardStep] | None], list[list[_State]]]:
    """正向 BFS：返回 状态 -> 父状态与步骤 的映射、逐层状态列表。"""
    visited: set[_State] = {start_state}
    layers: list[list[_State]] = [[start_state]]
    parents: dict[_State, tuple[_State, _ForwardStep] | None] = {start_state: None}
    for depth in range(max_depth):
        transitions: dict[_State, tuple[_State, _ForwardStep]] = {}
        for state in layers[depth]:
            if not budget.take():
                return parents, layers
            state_transitions = _forward_expand(state, rules, registry, visited)
            for new_state, step in state_transitions.items():
                if new_state not in transitions:
                    transitions[new_state] = (state, step)
        if not transitions:
            break
        ranked = sorted(
            transitions.items(),
            key=lambda item: (_distance(item[0], registry, [target]), len(item[0])),
        )[:beam_width]
        layer = [state for state, _ in ranked]
        for state, entry in ranked:
            visited.add(state)
            parents[state] = entry
        layers.append(layer)
    return parents, layers


# ------- 逆向搜索（反合成模板） -------


class _RetroTemplate:
    """反合成模板：从目标分子拆出候选反应物，并用对应正向规则验证。"""

    def __init__(
        self,
        name: str,
        forward_rule_name: str,
        generate: Callable[[oc.Molecule], list[list[oc.Molecule]]],
    ) -> None:
        self.name: str = name
        self.forward_rule_name: str = forward_rule_name
        self._generate: Callable[[oc.Molecule], list[list[oc.Molecule]]] = generate

    def generate(self, molecule: oc.Molecule) -> list[list[oc.Molecule]]:
        """对目标分子尝试拆解；返回候选反应物分子列表（每组均为含碳有机分子）。"""
        return self._generate(molecule)


def _validate_retro(
    target: oc.Molecule,
    reactants: Sequence[oc.Molecule],
    rule: rx.ReactionRule,
) -> tuple[rx.ReactionOutcome, tuple[oc.Molecule, ...]] | None:
    """用正向规则验证反合成拆解：产物含目标结构时返回 (命中, 实际输入)。"""
    reagent_groups = _REAGENT_INPUTS[rule.name]
    organic_indices = [index for index, keys in enumerate(reagent_groups) if not keys]
    if len(reactants) != len(organic_indices):
        return None
    choice_lists: list[tuple[str | None, ...]] = [
        tuple(keys) if keys else (None,) for keys in reagent_groups
    ]
    for choices in product(*choice_lists):
        inputs: list[oc.Molecule] = []
        combo_index = 0
        for index, keys in enumerate(reagent_groups):
            if keys:
                inputs.append(FREE_REAGENTS[choices[index]])
            else:
                inputs.append(reactants[combo_index])
                combo_index += 1
        try:
            outcomes = rx.reaction_outcomes(inputs, reaction=rule.name)
        except ValueError:
            continue
        for outcome in outcomes:
            if any(product == target for product in outcome.products):
                return outcome, tuple(inputs)
    return None


def _build_retro_templates() -> list[_RetroTemplate]:
    """构建 19 条反合成模板（均以对应正向规则做验证）。"""
    templates: list[_RetroTemplate] = []

    # 1. 烷烃 -> 烯烃（烯烃加氢的逆向）
    def gen_alkane_to_alkene(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for bond in m.bonds:
            a1, a2 = bond.atoms
            if bond.order != 1 or a1.name != "c" or a2.name != "c":
                continue
            if _in_pi(a1) or _in_pi(a2) or _has_multiple_bond(a1) or _has_multiple_bond(a2):
                continue
            if _total_h(a1) < 1 or _total_h(a2) < 1:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.add_bond(atom_map[a1], atom_map[a2])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("烷烃→烯烃", "烯烃加氢", gen_alkane_to_alkene))

    # 2. 烯烃 -> 炔烃（炔烃部分加氢的逆向）
    def gen_alkene_to_alkyne(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for bond in m.bonds:
            a1, a2 = bond.atoms
            if bond.order != 2 or a1.name != "c" or a2.name != "c":
                continue
            if _in_pi(a1) or _in_pi(a2):
                continue
            if _total_h(a1) < 1 or _total_h(a2) < 1:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.add_bond(atom_map[a1], atom_map[a2])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("烯烃→炔烃", "炔烃部分加氢", gen_alkene_to_alkyne))

    # 3. 烷烃 -> 炔烃（炔烃完全加氢的逆向）
    def gen_alkane_to_alkyne(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for bond in m.bonds:
            a1, a2 = bond.atoms
            if bond.order != 1 or a1.name != "c" or a2.name != "c":
                continue
            if _in_pi(a1) or _in_pi(a2) or _has_multiple_bond(a1) or _has_multiple_bond(a2):
                continue
            if _total_h(a1) < 1 or _total_h(a2) < 1:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.add_bond(atom_map[a1], atom_map[a2], order=2)
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("烷烃→炔烃", "炔烃完全加氢", gen_alkane_to_alkyne))

    # 4. 环己烷 -> 苯（苯加氢的逆向）
    def gen_cyclohexane_to_benzene(m: oc.Molecule) -> list[list[oc.Molecule]]:
        cycle = _find_carbon_cycle(m, 6)
        if cycle is None:
            return []
        ring_set = set(cycle)
        for atom in cycle:
            for bond in atom.bonds:
                neighbor = bond.other(atom)
                if neighbor in ring_set:
                    continue
                if neighbor.name != "h":
                    return []
        working, atom_map = _copy_with_map(m)
        try:
            oc.add_pi_system([atom_map[atom] for atom in cycle])
        except ValueError:
            return []
        components = _organic_components(working)
        return [components] if len(components) == 1 else []

    templates.append(_RetroTemplate("环己烷→苯", "苯加氢", gen_cyclohexane_to_benzene))

    # 5. 邻二卤代烃 -> 烯烃（烯烃与卤素加成的逆向）
    def gen_dihalide_to_alkene(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for bond in m.bonds:
            a1, a2 = bond.atoms
            if bond.order != 1 or a1.name != "c" or a2.name != "c":
                continue
            if _in_pi(a1) or _in_pi(a2) or _has_multiple_bond(a1) or _has_multiple_bond(a2):
                continue
            x1 = _single_halogen_neighbor(a1)
            x2 = _single_halogen_neighbor(a2)
            if x1 is None or x2 is None:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.del_atom(atom_map[x1])
                oc.del_atom(atom_map[x2])
                oc.add_bond(atom_map[a1], atom_map[a2])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("邻二卤代烃→烯烃", "烯烃与卤素加成", gen_dihalide_to_alkene))

    # 6. 卤代烃 -> 烯烃（烯烃与HX加成的逆向）
    def gen_haloalkane_to_alkene(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for bond in m.bonds:
            a1, a2 = bond.atoms
            if bond.order != 1 or a1.name != "c" or a2.name != "c":
                continue
            if _in_pi(a1) or _in_pi(a2) or _has_multiple_bond(a1) or _has_multiple_bond(a2):
                continue
            for halogen_carbon, x in (
                (a1, _single_halogen_neighbor(a1)),
                (a2, _single_halogen_neighbor(a2)),
            ):
                if x is None:
                    continue
                beta = a2 if halogen_carbon is a1 else a1
                if _total_h(beta) < 1:
                    continue
                working, atom_map = _copy_with_map(m)
                try:
                    oc.del_atom(atom_map[x])
                    oc.add_bond(atom_map[halogen_carbon], atom_map[beta])
                except ValueError:
                    continue
                components = _organic_components(working)
                if len(components) == 1:
                    results.append(components)
        return results

    templates.append(_RetroTemplate("卤代烃→烯烃", "烯烃与HX加成", gen_haloalkane_to_alkene))

    # 7. 醇 -> 烯烃（烯烃水化的逆向）
    def gen_alcohol_to_alkene(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for bond in m.bonds:
            a1, a2 = bond.atoms
            if bond.order != 1 or a1.name != "c" or a2.name != "c":
                continue
            if _in_pi(a1) or _in_pi(a2) or _has_multiple_bond(a1) or _has_multiple_bond(a2):
                continue
            for alcohol_carbon, o in (
                (a1, _single_o_neighbor(a1)),
                (a2, _single_o_neighbor(a2)),
            ):
                if o is None or _in_pi(o) or _is_carbonyl_carbon(alcohol_carbon):
                    continue
                beta = a2 if alcohol_carbon is a1 else a1
                if _total_h(beta) < 1:
                    continue
                working, atom_map = _copy_with_map(m)
                try:
                    oc.del_atom(atom_map[o])
                    oc.add_bond(atom_map[alcohol_carbon], atom_map[beta])
                except ValueError:
                    continue
                components = _organic_components(working)
                if len(components) == 1:
                    results.append(components)
        return results

    templates.append(_RetroTemplate("醇→烯烃", "烯烃水化", gen_alcohol_to_alkene))

    # 8. 对称醚 -> 2×同醇（醇分子间脱水的逆向）
    def gen_ether_to_alcohol(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for o in m.atoms:
            if o.name != "o" or _in_pi(o):
                continue
            c_neighbors = [n for n in _neighbors(o) if n.name == "c"]
            if len(c_neighbors) != 2:
                continue
            if any(
                _in_pi(c) or _has_multiple_bond(c) or _is_carbonyl_carbon(c)
                for c in c_neighbors
            ):
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.break_bond(atom_map[o], atom_map[c_neighbors[0]])
                oc.break_bond(atom_map[o], atom_map[c_neighbors[1]])
                oc.del_atom(atom_map[o])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 2 and components[0] == components[1]:
                results.append([components[0]])
        return results

    templates.append(_RetroTemplate("对称醚→2×同醇", "醇分子间脱水", gen_ether_to_alcohol))

    # 9. 2,4,6-三溴苯酚 -> 苯酚（苯酚与溴水的逆向）
    def gen_tribromophenol_to_phenol(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for pi in m.pi_systems:
            if len(pi.atoms) != 6 or any(a.name != "c" for a in pi.atoms):
                continue
            cycle = _ring_cycle(pi.atoms)
            if cycle is None:
                continue
            oh_carbon: oc.Atom | None = None
            for atom in cycle:
                for bond in atom.bonds:
                    neighbor = bond.other(atom)
                    if (
                        bond.order == 1
                        and neighbor.name == "o"
                        and neighbor not in cycle
                        and _total_h(neighbor) >= 1
                    ):
                        oh_carbon = atom
                        break
                if oh_carbon is not None:
                    break
            if oh_carbon is None:
                continue
            start = cycle.index(oh_carbon)
            rotated = cycle[start:] + cycle[:start]
            br_atoms: list[oc.Atom] = []
            valid = True
            for position in (1, 3, 5):
                candidates = [
                    n for n in _neighbors(rotated[position]) if _is_halogen(n)
                ]
                if len(candidates) != 1:
                    valid = False
                    break
                br_atoms.append(candidates[0])
            if not valid:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                for br in br_atoms:
                    oc.del_atom(atom_map[br])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("三溴苯酚→苯酚", "苯酚与溴水", gen_tribromophenol_to_phenol))

    # 10. 卤代烃 -> 烷烃（烷烃卤代的逆向）
    def gen_haloalkane_to_alkane(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "c" or _in_pi(atom) or _has_multiple_bond(atom):
                continue
            x = _single_halogen_neighbor(atom)
            if x is None:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.del_atom(atom_map[x])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("卤代烃→烷烃", "烷烃卤代", gen_haloalkane_to_alkane))

    # 11. 卤苯 -> 苯（苯卤代的逆向）
    def gen_halobenzene_to_benzene(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "c" or not _in_pi(atom):
                continue
            x = _single_halogen_neighbor(atom)
            if x is None:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.del_atom(atom_map[x])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("卤苯→苯", "苯卤代", gen_halobenzene_to_benzene))

    # 12. 硝基苯 -> 苯（苯硝化的逆向）
    def gen_nitrobenzene_to_benzene(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for pi_index, pi in enumerate(m.pi_systems):
            if not _is_nitro_pi(pi.atoms):
                continue
            n = next(atom for atom in pi.atoms if atom.name == "n")
            c_neighbors = [neighbor for neighbor in _neighbors(n) if neighbor.name == "c"]
            if len(c_neighbors) != 1 or not _in_pi(c_neighbors[0]):
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.remove_pi_system(working.pi_systems[pi_index])
                for atom in pi.atoms:
                    oc.del_atom(atom_map[atom])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("硝基苯→苯", "苯硝化", gen_nitrobenzene_to_benzene))

    # 13. 苯胺 -> 硝基苯（硝基还原的逆向）
    def gen_aniline_to_nitrobenzene(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "n" or _in_pi(atom) or _total_h(atom) < 1:
                continue
            heavy_neighbors = [n for n in _neighbors(atom) if n.name != "h"]
            if len(heavy_neighbors) != 1:
                continue
            c = heavy_neighbors[0]
            if c.name != "c" or not _in_pi(c):
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.del_atom(atom_map[atom])
                n2 = oc.Atom("n", working)
                o1 = oc.Atom("o", working)
                o2 = oc.Atom("o", working)
                oc.add_bond(atom_map[c], n2)
                oc.add_bond(n2, o1)
                oc.add_bond(n2, o2)
                oc.add_pi_system([n2, o1, o2])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("苯胺→硝基苯", "硝基还原", gen_aniline_to_nitrobenzene))

    # 14. 醛/酮 -> 醇（醇催化氧化的逆向）
    def gen_carbonyl_to_alcohol(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "c" or _has_single_o_neighbor(atom):
                continue
            double_o = [
                bond.other(atom)
                for bond in atom.bonds
                if bond.order == 2 and bond.other(atom).name == "o"
            ]
            if len(double_o) != 1:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.break_bond(atom_map[atom], atom_map[double_o[0]], order=1)
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("醛/酮→醇", "醇催化氧化", gen_carbonyl_to_alcohol))

    # 15. 羧酸 -> 醛（醛催化氧化的逆向）
    def gen_acid_to_aldehyde(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "c" or not _is_carbonyl_carbon(atom):
                continue
            double_o = [
                bond.other(atom)
                for bond in atom.bonds
                if bond.order == 2 and bond.other(atom).name == "o"
            ]
            if len(double_o) != 1:
                continue
            o_h = [
                neighbor
                for neighbor in _neighbors(atom)
                if neighbor.name == "o"
                and neighbor is not double_o[0]
                and _total_h(neighbor) >= 1
            ]
            if len(o_h) != 1:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.del_atom(atom_map[o_h[0]])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("羧酸→醛", "醛催化氧化", gen_acid_to_aldehyde))

    # 16. 伯醇 -> 醛（醛加氢还原的逆向）
    def gen_primary_alcohol_to_aldehyde(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "c" or _in_pi(atom) or _has_multiple_bond(atom):
                continue
            if _is_carbonyl_carbon(atom) or _total_h(atom) < 1:
                continue
            o = _single_o_neighbor(atom)
            if o is None or _in_pi(o) or _total_h(o) < 1:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.add_bond(atom_map[atom], atom_map[o])
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("伯醇→醛", "醛加氢还原", gen_primary_alcohol_to_aldehyde))

    # 17. 醇 -> 卤代烃（卤代烃水解的逆向，Cl/Br 两个变体）
    def gen_alcohol_to_haloalkane(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "c" or _in_pi(atom) or _has_multiple_bond(atom):
                continue
            if _is_carbonyl_carbon(atom):
                continue
            o = _single_o_neighbor(atom)
            if o is None or _in_pi(o) or _total_h(o) < 1:
                continue
            for x_name in ("cl", "br"):
                working, atom_map = _copy_with_map(m)
                try:
                    oc.del_atom(atom_map[o])
                    x = oc.Atom(x_name, working)
                    oc.add_bond(atom_map[atom], x)
                except ValueError:
                    continue
                components = _organic_components(working)
                if len(components) == 1:
                    results.append(components)
        return results

    templates.append(_RetroTemplate("醇→卤代烃", "卤代烃水解", gen_alcohol_to_haloalkane))

    # 18. 酯 -> 羧酸 + 醇（酯化反应的逆向）
    def gen_ester_to_acid_alcohol(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "c" or not _is_carbonyl_carbon(atom):
                continue
            double_o = [
                bond.other(atom)
                for bond in atom.bonds
                if bond.order == 2 and bond.other(atom).name == "o"
            ]
            if len(double_o) != 1:
                continue
            single_o = [
                bond.other(atom)
                for bond in atom.bonds
                if bond.order == 1 and bond.other(atom).name == "o"
            ]
            if len(single_o) != 1:
                continue
            bridge = single_o[0]
            c_rest = [
                neighbor
                for neighbor in _neighbors(bridge)
                if neighbor.name == "c" and neighbor is not atom
            ]
            if len(c_rest) != 1 or len(_neighbors(bridge)) != 2:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.break_bond(atom_map[atom], atom_map[bridge])
                oc.add_bond(atom_map[atom], oc.Atom("o", working))
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 2:
                results.append(components)
        return results

    templates.append(_RetroTemplate("酯→羧酸+醇", "酯化反应", gen_ester_to_acid_alcohol))

    # 19. 卤代烃 -> 醇（醇卤代的逆向）
    def gen_haloalkane_to_alcohol(m: oc.Molecule) -> list[list[oc.Molecule]]:
        results: list[list[oc.Molecule]] = []
        for atom in m.atoms:
            if atom.name != "c" or _in_pi(atom) or _has_multiple_bond(atom):
                continue
            if _is_carbonyl_carbon(atom):
                continue
            x = _single_halogen_neighbor(atom)
            if x is None:
                continue
            working, atom_map = _copy_with_map(m)
            try:
                oc.del_atom(atom_map[x])
                oc.add_bond(atom_map[atom], oc.Atom("o", working))
            except ValueError:
                continue
            components = _organic_components(working)
            if len(components) == 1:
                results.append(components)
        return results

    templates.append(_RetroTemplate("卤代烃→醇", "醇卤代", gen_haloalkane_to_alcohol))

    return templates


RETRO_TEMPLATES: tuple[_RetroTemplate, ...] = tuple(_build_retro_templates())


def _backward_expand(
    state: _State,
    templates: Sequence[_RetroTemplate],
    rules_by_name: dict[str, rx.ReactionRule],
    registry: dict[_Fp, oc.Molecule],
    visited: set[_State],
) -> dict[_State, tuple[_State, _BackwardStep]]:
    """对状态做一步逆向拆解：返回 新状态 -> (父状态, 拆解步骤)。"""
    transitions: dict[_State, tuple[_State, _BackwardStep]] = {}
    for key in sorted(state):
        molecule = registry[key]
        for template in templates:
            rule = rules_by_name[template.forward_rule_name]
            for generated in template.generate(molecule):
                # 先换成仓库中的命名实例再验证：结构判等不变，步骤输入保留原名
                reactants = [_resolve_named(item, registry) for item in generated]
                validated = _validate_retro(molecule, reactants, rule)
                if validated is None:
                    continue
                outcome, inputs = validated
                new_reactants: set[_Fp] = set()
                for reactant in reactants:
                    fp = _fp(reactant)
                    registry.setdefault(fp, reactant)
                    new_reactants.add(fp)
                new_state = (state - {key}) | frozenset(new_reactants)
                if new_state == state or new_state in visited:
                    continue
                if new_state not in transitions:
                    transitions[new_state] = (
                        state,
                        _BackwardStep(rule, inputs, outcome),
                    )
    return transitions


def _backward_layers(
    start_state: _State,
    templates: Sequence[_RetroTemplate],
    rules_by_name: dict[str, rx.ReactionRule],
    registry: dict[_Fp, oc.Molecule],
    reference: Sequence[oc.Molecule],
    max_depth: int,
    beam_width: int,
    budget: _Budget,
) -> tuple[dict[_State, tuple[_State, _BackwardStep] | None], list[list[_State]]]:
    """逆向 BFS：返回 状态 -> 父状态与拆解步骤 的映射、逐层状态列表。"""
    visited: set[_State] = {start_state}
    layers: list[list[_State]] = [[start_state]]
    parents: dict[_State, tuple[_State, _BackwardStep] | None] = {start_state: None}
    for depth in range(max_depth):
        transitions: dict[_State, tuple[_State, _BackwardStep]] = {}
        for state in layers[depth]:
            if not budget.take():
                return parents, layers
            state_transitions = _backward_expand(state, templates, rules_by_name, registry, visited)
            for new_state, entry in state_transitions.items():
                if new_state not in transitions:
                    transitions[new_state] = entry
        if not transitions:
            break
        ranked = sorted(
            transitions.items(),
            key=lambda item: (_distance(item[0], registry, reference), len(item[0])),
        )[:beam_width]
        layer = [state for state, _ in ranked]
        for state, entry in ranked:
            visited.add(state)
            parents[state] = entry
        layers.append(layer)
    return parents, layers


# ------- 汇合与路线重建 -------


def _forward_to_step(step: _ForwardStep) -> SynthesisStep:
    return SynthesisStep(
        step_no=0,
        rule_name=step.rule.name,
        category=step.rule.category,
        conditions=step.rule.conditions,
        inputs=step.inputs,
        outputs=tuple(step.outcome.products),
    )


def _backward_to_step(step: _BackwardStep) -> SynthesisStep:
    return SynthesisStep(
        step_no=0,
        rule_name=step.rule.name,
        category=step.rule.category,
        conditions=step.rule.conditions,
        inputs=step.inputs,
        outputs=tuple(step.outcome.products),
    )


def _build_route(
    target: oc.Molecule,
    f_state: _State,
    b_state: _State,
    f_parents: dict[_State, tuple[_State, _ForwardStep] | None],
    b_parents: dict[_State, tuple[_State, _BackwardStep] | None],
) -> SynthesisRoute:
    """由汇合状态重建完整路线：正向路径 + 逆向路径反向。"""
    forward_steps: list[_ForwardStep] = []
    state = f_state
    while f_parents[state] is not None:
        parent, step = f_parents[state]
        assert step is not None
        forward_steps.append(step)
        state = parent
    forward_steps.reverse()

    backward_steps: list[_BackwardStep] = []
    state = b_state
    while b_parents[state] is not None:
        parent, step = b_parents[state]
        assert step is not None
        backward_steps.append(step)
        state = parent
    backward_steps.reverse()

    steps: list[SynthesisStep] = [_forward_to_step(step) for step in forward_steps]
    steps.extend(_backward_to_step(step) for step in reversed(backward_steps))
    for index, step in enumerate(steps, start=1):
        steps[index - 1] = SynthesisStep(
            step_no=index,
            rule_name=step.rule_name,
            category=step.category,
            conditions=step.conditions,
            inputs=step.inputs,
            outputs=step.outputs,
        )
    return SynthesisRoute(target=target, steps=tuple(steps))


def _route_key(route: SynthesisRoute) -> tuple[tuple[str, tuple[_Fp, ...], tuple[_Fp, ...]], ...]:
    return tuple(
        (
            step.rule_name,
            tuple(_fp(molecule) for molecule in step.inputs),
            tuple(_fp(molecule) for molecule in step.outputs),
        )
        for step in route.steps
    )


def _organic_fps(molecules: Sequence[oc.Molecule]) -> set[_Fp]:
    return {_fp(molecule) for molecule in molecules if _has_carbon(molecule)}


def _minimize_route(route: SynthesisRoute, start: Sequence[oc.Molecule]) -> SynthesisRoute:
    """删除冗余步骤：某步的所有输入在更早状态已可获得、去掉后目标仍可达时移除。

    集合语义下，状态只增不减，正向搜索可能拼出“先用掉、再复用起始物”的
    绕路步骤；本函数按链式可用性收紧，保证返回的路线每一步都必要。
    """
    steps = list(route.steps)
    start_fps = {_fp(molecule) for molecule in start}

    def valid_without(index: int) -> bool:
        available = set(start_fps)
        for step_index, step in enumerate(steps):
            if step_index == index:
                continue
            if not _organic_fps(step.inputs) <= available:
                return False
            available |= _organic_fps(step.outputs)
        return _fp(route.target) in available

    changed = True
    while changed:
        changed = False
        for index in range(len(steps)):
            if valid_without(index):
                del steps[index]
                changed = True
                break

    numbered: list[SynthesisStep] = []
    for index, step in enumerate(steps, start=1):
        numbered.append(
            SynthesisStep(
                step_no=index,
                rule_name=step.rule_name,
                category=step.category,
                conditions=step.conditions,
                inputs=step.inputs,
                outputs=step.outputs,
            )
        )
    return SynthesisRoute(target=route.target, steps=tuple(numbered))


def _strategy_signature(route: SynthesisRoute) -> tuple[str, ...]:
    """路线的思路签名：各步反应规则名的排序元组（集合口径）。

    忽略步骤顺序、中间体与试剂：两条路线用到的反应原理集合相同，
    即视为同一思路的变体（例如「先卤代再取代」与「先取代再卤代」）。
    """
    return tuple(sorted(step.rule_name for step in route.steps))


def _filter_routes(
    routes: list[SynthesisRoute],
    *,
    dedupe_strategy: bool,
    optimal_only: bool,
) -> list[SynthesisRoute]:
    """按思路与步数精简路线列表（保持传入顺序，调用方需已按步数升序排好）。

    - dedupe_strategy：同一思路签名只保留第一条，即该思路的最短代表；
    - optimal_only：在去重之后只保留步数等于全局最小步数的路线。
    """
    if dedupe_strategy:
        seen: set[tuple[str, ...]] = set()
        deduped: list[SynthesisRoute] = []
        for route in routes:
            signature = _strategy_signature(route)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(route)
        routes = deduped
    if optimal_only and routes:
        best_steps = min(route.step_count for route in routes)
        routes = [route for route in routes if route.step_count == best_steps]
    return routes


# ------- 公共 API -------


def plan_synthesis(
    reactants: Sequence[oc.Molecule],
    target: oc.Molecule,
    *,
    reaction: str | None = None,
    conditions: str | None = None,
    category: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    max_routes: int | None = DEFAULT_MAX_ROUTES,
    dedupe_strategy: bool = False,
    optimal_only: bool = False,
) -> list[SynthesisRoute]:
    """规划从起始反应物到目标产物的合成路线（默认最多 6 步）。

    - reactants：含碳有机起始反应物（小分子试剂由系统自动提供，无需输入）；
    - target：含碳有机目标产物；
    - reaction / conditions / category：按 oc_reactions.find_reactions 语义
      硬性筛选可用规则；筛选后无规则抛 ValueError；
    - dedupe_strategy：按思路签名（各步反应规则名的集合）去重，每种思路
      只保留最短代表；默认 False；
    - optimal_only：只保留步数最少的路线；默认 False。
    - 返回路线按总步数升序、按结构去重，最多 max_routes 条；找不到返回空列表。
      两个过滤均在排序之后、max_routes 截断之前生效，截断的是过滤后的结果。
    """
    if not isinstance(reactants, (list, tuple)):
        raise ValueError("reactants 必须是 Molecule 列表或元组")
    reactant_list = list(reactants)
    if not reactant_list:
        raise ValueError("至少需要一个含碳有机反应物")
    for molecule in reactant_list:
        if not isinstance(molecule, oc.Molecule):
            raise ValueError("反应物必须是 organic_chemistry.Molecule 实例")
        if not _has_carbon(molecule):
            raise ValueError("起始反应物必须为含碳有机分子（小分子试剂由系统自动提供）")
    if not isinstance(target, oc.Molecule):
        raise ValueError("target 必须是 organic_chemistry.Molecule 实例")
    if not _has_carbon(target):
        raise ValueError("目标产物必须为含碳有机分子")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps 必须为正整数")
    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width 必须为正整数")
    if max_routes is not None and (not isinstance(max_routes, int) or max_routes < 1):
        raise ValueError("max_routes 必须为正整数或 None")
    if not isinstance(dedupe_strategy, bool):
        raise ValueError("dedupe_strategy 必须为布尔值")
    if not isinstance(optimal_only, bool):
        raise ValueError("optimal_only 必须为布尔值")

    if any(molecule == target for molecule in reactant_list):
        return [SynthesisRoute(target=target, steps=())]

    rules = rx.find_reactions([], reaction=reaction, conditions=conditions, category=category)
    if not rules:
        raise ValueError("未找到符合反应名/条件/类别的反应规则")
    rules_by_name = {rule.name: rule for rule in rules}

    registry: dict[_Fp, oc.Molecule] = {}
    for molecule in [target, *reactant_list]:
        registry.setdefault(_fp(molecule), molecule)
    start_state = frozenset(_fp(molecule) for molecule in reactant_list)
    target_state = frozenset({_fp(target)})

    templates = [
        template for template in RETRO_TEMPLATES
        if template.forward_rule_name in rules_by_name
    ]
    max_forward = max_steps // 2
    max_backward = max_steps - max_forward

    budget = _Budget(DEFAULT_NODE_BUDGET)
    f_parents, f_layers = _forward_layers(
        start_state, target, registry, rules, max_forward, beam_width, budget
    )
    b_parents, b_layers = _backward_layers(
        target_state,
        templates,
        rules_by_name,
        registry,
        reactant_list,
        max_backward,
        beam_width,
        budget,
    )

    routes: list[SynthesisRoute] = []
    route_keys: set[tuple] = set()
    for f_layer in f_layers:
        for f_state in f_layer:
            for b_layer in b_layers:
                for b_state in b_layer:
                    if len(b_state) > len(f_state) or not b_state <= f_state:
                        continue
                    route = _build_route(target, f_state, b_state, f_parents, b_parents)
                    route = _minimize_route(route, reactant_list)
                    key = _route_key(route)
                    if key in route_keys:
                        continue
                    route_keys.add(key)
                    routes.append(route)

    routes.sort(key=lambda route: (route.step_count, _route_key(route)))
    routes = _filter_routes(
        routes, dedupe_strategy=dedupe_strategy, optimal_only=optimal_only
    )
    if max_routes is not None:
        routes = routes[:max_routes]
    return routes


def _display_formula(formula: dict[str, int]) -> str:
    parts: list[str] = []
    for element, count in formula.items():
        if count == 0:
            continue
        parts.append(element.capitalize() + (str(count) if count != 1 else ""))
    return "".join(parts)


def _describe(molecule: oc.Molecule) -> str:
    formula = _display_formula(molecule.formula)
    if molecule.name:
        return f"{molecule.name}({formula})"
    return formula


def _describe_list(molecules: Sequence[oc.Molecule]) -> str:
    return " + ".join(_describe(molecule) for molecule in molecules)


def synthesis_report(route: SynthesisRoute) -> str:
    """生成路线的中文文字报告。"""
    lines = [
        f"合成路线（共 {route.step_count} 步）",
        f"目标产物：{_describe(route.target)}",
    ]
    for step in route.steps:
        condition = step.conditions if step.conditions else "—"
        lines.append(f"{step.step_no}. {step.rule_name} [{step.category}] 条件：{condition}")
        lines.append(f"   {_describe_list(step.inputs)} → {_describe_list(step.outputs)}")
    return "\n".join(lines)


# ------- 配置自检 -------


def _check_config() -> None:
    rules_by_name = {rule.name: rule for rule in rx.list_reactions()}
    if set(_REAGENT_INPUTS) != set(rules_by_name):
        raise RuntimeError("试剂映射与反应规则目录不一致")
    for name, groups in _REAGENT_INPUTS.items():
        rule = rules_by_name[name]
        if len(groups) != rule.required_inputs:
            raise RuntimeError(f"规则 {name} 的试剂映射长度与输入数不一致")
        for keys in groups:
            for key in keys:
                if key not in FREE_REAGENTS:
                    raise RuntimeError(f"规则 {name} 引用了未知试剂 {key}")
    for template in RETRO_TEMPLATES:
        if template.forward_rule_name not in rules_by_name:
            raise RuntimeError(
                f"反合成模板 {template.name} 引用了未知规则 {template.forward_rule_name}"
            )


_check_config()


if __name__ == "__main__":
    # 演示：苯 -> 苯胺、乙醇 -> 乙酸乙酯
    def _benzene() -> oc.Molecule:
        molecule = oc.Molecule(name="苯")
        cs = [oc.Atom("c", molecule) for _ in range(6)]
        for i in range(6):
            oc.add_bond(cs[i], cs[(i + 1) % 6])
        oc.add_pi_system(cs)
        return molecule

    def _aniline() -> oc.Molecule:
        molecule = _benzene()
        oc.add_bond(molecule.atoms[0], oc.Atom("n", molecule))
        molecule.name = "苯胺"
        return molecule

    def _chain(n: int) -> oc.Molecule:
        molecule = oc.Molecule()
        cs = [oc.Atom("c", molecule) for _ in range(n)]
        for i in range(n - 1):
            oc.add_bond(cs[i], cs[i + 1])
        return molecule

    def _ethanol() -> oc.Molecule:
        molecule = _chain(2)
        oc.add_bond(molecule.atoms[1], oc.Atom("o", molecule))
        return molecule

    def _ethyl_acetate() -> oc.Molecule:
        molecule = oc.Molecule(name="乙酸乙酯")
        c1, c2, c3, c4 = (oc.Atom("c", molecule) for _ in range(4))
        o_d = oc.Atom("o", molecule)
        o_b = oc.Atom("o", molecule)
        oc.add_bond(c1, c2)
        oc.add_bond(c2, o_d, order=2)
        oc.add_bond(c2, o_b)
        oc.add_bond(o_b, c3)
        oc.add_bond(c3, c4)
        return molecule

    routes = plan_synthesis([_benzene()], _aniline())
    print(synthesis_report(routes[0]) if routes else "未找到路线")
    print()
    routes = plan_synthesis([_ethanol()], _ethyl_acetate())
    print(synthesis_report(routes[0]) if routes else "未找到路线")
