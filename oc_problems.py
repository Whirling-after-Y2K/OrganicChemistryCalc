"""反应结果 × 同分异构体联动搜索（无第三方依赖）。

把 oc_reactions（模拟反应）与 oc_isomers（异构体枚举）联动：
输入目标分子式 + 反应筛选 + 产物约束，自动枚举该式的反应物异构体、
逐个模拟反应，返回产物满足约束的反应物及其反应结果。

约束语义：
- ProductConstraint 可同时携带等位氢组数、等位氢约分比模式、必含基团
  三个条件，同一产物需同时满足（"与"关系）；
- 约束与含碳有机产物之间强制一一配对（单射）：每条约束必须由不同产物满足；
- 小分子产物（HX、H2O 等不含碳）不参与约束匹配。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import oc_isomers
import oc_reactions as rx
import organic_chemistry as oc


@dataclass(frozen=True)
class ProductConstraint:
    """产物约束：等位氢组数 / 等位氢约分比模式 / 必含基团（可多选，全部满足）。"""

    equivalent_h_groups: int | None = None
    equivalent_hydrogens: Sequence[int] | None = None
    required_groups: Sequence[str] | None = None


def _validate_constraint(constraint: ProductConstraint) -> None:
    if (
        constraint.equivalent_h_groups is None
        and constraint.equivalent_hydrogens is None
        and constraint.required_groups is None
    ):
        raise ValueError("产物约束至少需要指定一个条件（组数/约分比/基团）")
    if constraint.equivalent_h_groups is not None:
        if (
            not isinstance(constraint.equivalent_h_groups, int)
            or constraint.equivalent_h_groups <= 0
        ):
            raise ValueError("等位氢组数必须为正整数")
    if constraint.equivalent_hydrogens is not None:
        values = list(constraint.equivalent_hydrogens)
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("等位氢约分比模式必须为正整数序列")
    if constraint.required_groups is not None:
        # 校验基团名称（未知名称抛 ValueError）
        oc_isomers._resolve_required_groups(constraint.required_groups)


def _organic_products(products: Sequence[oc.Molecule]) -> list[oc.Molecule]:
    """含碳有机产物（小分子 HX/H2O 不参与约束匹配）。"""
    return [product for product in products if product.formula.get('c', 0) > 0]


def _satisfies(product: oc.Molecule, constraint: ProductConstraint) -> bool:
    """单个产物是否满足单条约束的全部条件（与关系）。"""
    if constraint.equivalent_h_groups is not None:
        if len(product.equivalent_hydrogen_groups) != constraint.equivalent_h_groups:
            return False
    if constraint.equivalent_hydrogens is not None:
        if (
            oc_isomers._reduce_hydrogen_pattern(product.equivalent_hydrogen_groups)
            != oc_isomers._reduce_hydrogen_pattern(constraint.equivalent_hydrogens)
        ):
            return False
    if constraint.required_groups is not None:
        resolved = oc_isomers._resolve_required_groups(constraint.required_groups)
        if not oc_isomers._contains_groups(product, resolved):
            return False
    return True


def matches_product_constraints(
    products: Sequence[oc.Molecule],
    constraints: Sequence[ProductConstraint],
) -> bool:
    """约束与含碳产物之间是否存在单射匹配（每条约束由不同产物满足）。

    空约束序列返回 True；约束数大于含碳产物数时直接返回 False。
    """
    constraints = list(constraints)
    for constraint in constraints:
        _validate_constraint(constraint)
    if not constraints:
        return True
    candidates = _organic_products(products)
    if len(constraints) > len(candidates):
        return False

    def backtrack(index: int, used: set[int]) -> bool:
        if index == len(constraints):
            return True
        constraint = constraints[index]
        for p_index, product in enumerate(candidates):
            if p_index in used:
                continue
            if _satisfies(product, constraint):
                used.add(p_index)
                if backtrack(index + 1, used):
                    return True
                used.remove(p_index)
        return False

    return backtrack(0, set())


def find_reactants(
    formula: oc.Molecule,
    product_constraints: Sequence[ProductConstraint],
    *,
    reagents: Sequence[oc.Molecule] = (),
    reaction: str | None = None,
    conditions: str | None = None,
    category: str | None = None,
    reactant_groups: Sequence[str] | None = None,
    reactant_hydrogens: Sequence[int] | None = None,
) -> list[tuple[oc.Molecule, list[rx.ReactionOutcome]]]:
    """搜索目标分子式的反应物异构体，其反应产物满足全部产物约束。

    - 候选反应物 = oc_isomers.find_isomers(formula, required_groups=reactant_groups,
      equivalent_hydrogens=reactant_hydrogens)；
    - 对每个候选模拟 reaction_outcomes([候选, *reagents], reaction/conditions/
      category)，保留产物约束匹配成功的结果；规则不匹配或反应物数量不符的
      候选静默跳过；
    - 反应名/条件筛选不出任何规则时抛 ValueError；
    - 返回 (反应物, 命中结果列表) 列表，顺序与候选枚举一致。
    """
    constraints = list(product_constraints)
    for constraint in constraints:
        _validate_constraint(constraint)

    rules = rx.find_reactions(
        [formula],
        reaction=reaction,
        conditions=conditions,
        category=category,
    )
    if not rules:
        raise ValueError("未找到符合反应名/条件/类别的反应规则")

    candidates = oc_isomers.find_isomers(
        formula,
        required_groups=reactant_groups,
        equivalent_hydrogens=reactant_hydrogens,
    )
    results: list[tuple[oc.Molecule, list[rx.ReactionOutcome]]] = []
    for candidate in candidates:
        try:
            outcomes = rx.reaction_outcomes(
                [candidate, *reagents],
                reaction=reaction,
                conditions=conditions,
                category=category,
            )
        except ValueError:
            continue
        matched = [
            outcome
            for outcome in outcomes
            if matches_product_constraints(outcome.products, constraints)
        ]
        if matched:
            results.append((candidate, matched))
    return results


if __name__ == "__main__":
    # 例：某酯 C5H10O2 水解后，一产物有 4 种等位氢，另一产物含羟基
    def formula_molecule() -> oc.Molecule:
        m = oc.Molecule()
        carbonyl = oc.Atom('c', m)
        c2 = oc.Atom('c', m)
        oc.add_bond(carbonyl, c2)
        oc.add_bond(c2, oc.Atom('c', m))
        oc.add_bond(carbonyl, oc.Atom('o', m), 2)
        bridge = oc.Atom('o', m)
        oc.add_bond(carbonyl, bridge)
        ch2 = oc.Atom('c', m)
        oc.add_bond(bridge, ch2)
        oc.add_bond(ch2, oc.Atom('c', m))
        return m

    def water() -> oc.Molecule:
        m = oc.Molecule()
        oc.Atom('o', m)
        return m

    found = find_reactants(
        formula_molecule(),
        [
            ProductConstraint(equivalent_h_groups=4),
            ProductConstraint(required_groups=["羟基"]),
        ],
        reagents=[water()],
        reaction="水解",
        reactant_groups=["酯基"],
    )
    print(f"命中反应物 {len(found)} 个：")
    for reactant, outcomes in found:
        print("-", reactant.formula)
        for outcome in outcomes:
            products = "、".join(
                f"{p.formula}(等位氢{p.equivalent_hydrogen_groups})"
                for p in outcome.products
            )
            print(f"  反应 {outcome.rule.name} → {products}")
