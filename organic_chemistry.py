"""
有机化学核心数据结构（半显式原子模型）

设计原则：
- 分子图（原子 Atom、键 Bond、π 体系 PiSystem）是唯一数据源，
  分子式、不饱和度等派生数据读取时现算，不再手工维护副本。
- F/Cl/Br/I 等单价元素是真正的图节点；H 默认由自由价隐式推导，
  需要表示特殊活性（如酸性氢）时使用显式节点 ActiveH。
- 结构相等：先以 Weisfeiler-Lehman 迭代哈希指纹快照做预筛，
  再以自研回溯同构确认做精确判等（WL 预筛可能存在假阳性，
  但不会把真正相等的结构判为不等）。采用严格判等：
  普通显式 H 与隐氢视为不同结构，活性 H 参与指纹。
- 指纹快照契约：结构指纹是快照，编辑完成后必须调用 Molecule.update()
  重新计算；未 update() 时读取 feature 或做 == 判等会抛 ValueError。
  直接修改 bond.order / atom.charge / pi.dbe 等标量字段属契约外操作。
- 表示公约：显式多重键仅表示局域键；同一 π 体系成员之间只允许单键，
  离域体系一律用 add_pi_bond 创建 PiSystem 表示，不得用多重键代替。
  违反该公约视为无效结构，在编辑、validate() 与派生性质计算时抛 ValueError。
"""

from __future__ import annotations

import hashlib
from typing import Literal, TypeAlias, cast

# ------- 常量 -------

# 元素 -> 价键数；表中的元素都可作为显式图节点
ElementName: TypeAlias = Literal["c", "n", "o", "h", "f", "cl", "br", "i"]
CHEMISTRY_BOND_DICT: dict[ElementName, int] = {
    "c": 4,
    "n": 3,
    "o": 2,
    "h": 1,
    "f": 1,
    "cl": 1,
    "br": 1,
    "i": 1,
}
MAX_BOND_ORDER: int = 3

# 指纹迭代中的原子标签：初值为元组，细化后收敛为定长十六进制串
Label: TypeAlias = tuple[object, ...]
AtomLabel: TypeAlias = Label | str


# ------- 数据结构 -------

class Molecule:
    """分子：原子、键、π 体系的容器，以及所有派生性质的现算入口。"""

    def __init__(self, name: str | None = None) -> None:
        self.name: str | None = name
        self.atoms: list[Atom] = []
        self.bonds: list[Bond] = []
        self.pi_systems: list[PiSystem] = []
        self._feature_cache: list[str] | None = None  # update() 后生效的指纹快照

    # ---- 派生数据（全部现算，不存储） ----

    @property
    def formula(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.name] = counts.get(atom.name, 0) + 1
        counts['h'] = counts.get('h', 0) + sum(atom.implicit_h for atom in self.atoms)
        priority = {'c': 0, 'h': 1}
        return dict(sorted(counts.items(), key=lambda item: (priority.get(item[0], 2), item[0])))

    @property
    def component_count(self) -> int:
        """连通分量数（BFS）。"""
        seen: set[Atom] = set()
        count = 0
        for atom in self.atoms:
            if atom in seen:
                continue
            count += 1
            stack: list[Atom] = [atom]
            seen.add(atom)
            while stack:
                current = stack.pop()
                for bond in current.bonds:
                    neighbor = bond.other(current)
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        return count

    @property
    def ring_count(self) -> int:
        """环数 = 键数 - 原子数 + 连通分量数。"""
        if not self.atoms:
            return 0
        return len(self.bonds) - len(self.atoms) + self.component_count

    @property
    def unsaturation(self) -> int:
        """不饱和度 = 键级超出部分 + 环数 + π 体系贡献。

        计算前先校验结构（validate），无效结构抛 ValueError，
        避免静默给出错误数值。
        """
        self.validate()
        bond_part = sum(bond.order - 1 for bond in self.bonds)
        return bond_part + self.ring_count + sum(pi.dbe for pi in self.pi_systems)

    def validate(self) -> None:
        """校验结构不变量；无效结构抛 ValueError。

        检查范围：
        - 容器一致性：molecule.bonds 与各 atom.bonds 双向一致、
          无重复、atom.belong 正确、键与 π 成员属于分子；
        - 键自身：1 <= order <= MAX_BOND_ORDER，无自环；
        - 价键：每个原子的 used_valence 不超过其价键数；
        - π 体系：成员 >= 2、无重复、连通、成员间无显式多重键、
          每原子至多参与一个 π 体系、单价元素不参与。
        update() 与 unsaturation 计算前会自动调用。
        """
        _check_containers(self)
        for atom in self.atoms:
            _check_valence(atom, 0)
        for bond in self.bonds:
            if bond.atoms[0] is bond.atoms[1]:
                raise ValueError("原子不能与自身成键")
            if not 1 <= bond.order <= MAX_BOND_ORDER:
                raise ValueError(f"键级必须为 1-{MAX_BOND_ORDER}")
        for pi in self.pi_systems:
            _check_pi_system(pi)
        participation: dict[Atom, int] = {}
        for pi in self.pi_systems:
            for atom in pi.atoms:
                participation[atom] = participation.get(atom, 0) + 1
                if participation[atom] > 1:
                    raise ValueError(f"{atom.name} 原子已参与多个 π 体系")

    def update(self) -> None:
        """校验结构并重算结构指纹快照。

        编辑完成后必须调用；未 update() 时读取 feature 或做 == 判等
        会抛 ValueError。直接修改 bond.order / atom.charge / pi.dbe
        等标量字段属契约外操作，修改后同样需要重新 update()。
        """
        self.validate()
        self._feature_cache = _fingerprint(self)

    @property
    def feature(self) -> list[str]:
        """结构指纹快照（与建键顺序无关），update() 后可用。

        返回防御性拷贝：外部修改返回值不影响内部缓存。
        """
        if self._feature_cache is None:
            raise ValueError("请先调用 update()")
        return list(self._feature_cache)

    # ---- 内置方法 ----

    def __eq__(self, other: object) -> bool:
        """精确结构判等：WL 指纹快照预筛 + 回溯同构确认。

        双方都必须先调用 update()，否则抛 ValueError。
        """
        if not isinstance(other, Molecule):
            return False
        if self.feature != other.feature:
            return False
        return _are_isomorphic(self, other)

    # 注意：Molecule 定义了 __eq__ 且结构可变，因此不定义 __hash__，
    # Python 会自动令其实例不可哈希；需要作集合/字典键时用 tuple(molecule.feature)。
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Molecule({_display_formula(self.formula)})"


class Atom:
    """原子：只记录身份与连接关系，特征全部派生。"""

    def __init__(self, name: str, molecule: Molecule) -> None:
        name = name.lower()
        if name not in CHEMISTRY_BOND_DICT:
            raise ValueError(f"{name} 不是可成键元素：{sorted(CHEMISTRY_BOND_DICT)}")
        self.name: ElementName = cast(ElementName, name)
        self.bonds: list[Bond] = []   # 参与的所有键（Bond 对象）
        self.charge: int = 0          # 形式电荷（预留，暂不参与指纹）
        self.belong: Molecule = molecule
        molecule.atoms.append(self)

    @property
    def used_valence(self) -> int:
        """已占用的价键数 = 键级之和 + π 体系槽位数。"""
        used = sum(bond.order for bond in self.bonds)
        used += sum(1 for pi in self.belong.pi_systems if self in pi.atoms)
        return used

    @property
    def implicit_h(self) -> int:
        """隐氢数 = 价键数 - 已占用价键数；单价元素不计隐氢（空槽位为自由基）。"""
        if CHEMISTRY_BOND_DICT[self.name] < 2:
            return 0
        return max(0, CHEMISTRY_BOND_DICT[self.name] - self.used_valence)

    def __repr__(self) -> str:
        return f"<Atom {self.name}>"


class ActiveH(Atom):
    """具有特殊活性的显式氢原子（如酸性氢）。

    作为图上的真实节点参与建键、删键与指纹判等，
    与普通隐氢、普通显式氢均不同。
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__('h', molecule)


class Bond:
    """键：一等对象，两个端点 + 键级 + 预留的立体化学字段。"""

    def __init__(
        self,
        atom1: Atom,
        atom2: Atom,
        order: int = 1,
        stereo: str | None = None,
        aromatic: bool = False,
    ) -> None:
        if atom1 is atom2:
            raise ValueError("原子不能与自身成键")
        if atom1.belong is not atom2.belong:
            raise ValueError("不能连接不同分子的原子")
        if not 1 <= order <= MAX_BOND_ORDER:
            raise ValueError(f"键级必须为 1-{MAX_BOND_ORDER}")
        self.atoms: tuple[Atom, Atom] = (atom1, atom2)
        self.order: int = order
        self.stereo: str | None = stereo   # 预留：顺反异构/立体构型
        self.aromatic: bool = aromatic     # 预留：是否为芳香键
        atom1.bonds.append(self)
        atom2.bonds.append(self)
        atom1.belong.bonds.append(self)

    def other(self, atom: Atom) -> Atom:
        """返回键的另一端原子。"""
        atom1, atom2 = self.atoms
        return atom2 if atom is atom1 else atom1

    def __repr__(self) -> str:
        atom1, atom2 = self.atoms
        return f"<Bond {atom1.name}-{atom2.name} order={self.order}>"


class PiSystem:
    """离域 π 体系：一组原子 + 不饱和度贡献（dbe）。

    表示公约：离域体系一律用 PiSystem 表示，成员之间只允许单键；
    显式多重键仅表示局域键，不得与 π 体系混用于同一对原子。
    """

    def __init__(
        self,
        atoms: list[Atom],
        dbe: int | None = None,
        aromatic: bool = False,
    ) -> None:
        if len(atoms) < 2:
            raise ValueError("π 体系至少需要两个原子")
        self.atoms: list[Atom] = list(atoms)
        self.dbe: int = _infer_pi_dbe(self.atoms) if dbe is None else dbe
        self.aromatic: bool = aromatic

    def __repr__(self) -> str:
        names = ','.join(atom.name for atom in self.atoms)
        return f"<PiSystem({names}) dbe={self.dbe}>"


# ------- 构建与编辑操作 -------

def _find_bond(atom1: Atom, atom2: Atom) -> Bond | None:
    for bond in atom1.bonds:
        if bond.other(atom1) is atom2:
            return bond
    return None


def _check_valence(atom: Atom, delta: int) -> None:
    used = atom.used_valence + delta
    limit = CHEMISTRY_BOND_DICT[atom.name]
    if used > limit:
        raise ValueError(f"{atom.name} 原子价键数不足：需要 {used}，最多 {limit}")


def _in_same_pi_system(atom1: Atom, atom2: Atom, molecule: Molecule) -> bool:
    """atom1 与 atom2 是否同属于某个 π 体系。"""
    return any(atom1 in pi.atoms and atom2 in pi.atoms for pi in molecule.pi_systems)


def _check_multiple_bond_within_pi(atom1: Atom, atom2: Atom, order: int) -> None:
    """表示公约：同一 π 体系成员之间只允许单键。"""
    if order > 1 and _in_same_pi_system(atom1, atom2, atom1.belong):
        raise ValueError("π 体系成员之间不能画显式多重键")


def add_bond(atom1: Atom, atom2: Atom, order: int = 1) -> Bond:
    """在 atom1 与 atom2 之间建立键；若已存在则提升键级。

    表示公约：同一 π 体系成员之间只允许单键；离域体系请用 add_pi_bond，
    不要用显式多重键代替。
    """
    bond = _find_bond(atom1, atom2)
    if bond is not None:
        new_order = bond.order + order
        if new_order > MAX_BOND_ORDER:
            raise ValueError(f"键级不能超过 {MAX_BOND_ORDER}")
        _check_multiple_bond_within_pi(atom1, atom2, new_order)
        _check_valence(atom1, order)
        _check_valence(atom2, order)
        bond.order = new_order
        return bond
    _check_multiple_bond_within_pi(atom1, atom2, order)
    _check_valence(atom1, order)
    _check_valence(atom2, order)
    return Bond(atom1, atom2, order)


def break_bond(atom1: Atom, atom2: Atom, order: int = 0) -> None:
    """断开 atom1 与 atom2 之间的键。

    order 为 0（默认）时整根断开；order > 0 时降低 N 个键级，
    若不足以降到 1 则整根断开；order < 0 抛 ValueError。
    """
    if order < 0:
        raise ValueError("order 必须 >= 0")
    bond = _find_bond(atom1, atom2)
    if bond is None:
        raise ValueError("两个原子之间不存在键")
    if order > 0 and bond.order - order >= 1:
        bond.order -= order
        return
    molecule = atom1.belong
    for atom in bond.atoms:
        atom.bonds.remove(bond)
    molecule.bonds.remove(bond)


def _pi_members_connected(atom_list: list[Atom]) -> bool:
    """π 体系成员是否通过键彼此连通（诱导子图连通），且无重复成员。"""
    members = set(atom_list)
    if len(members) != len(atom_list):
        return False
    visited: set[Atom] = {atom_list[0]}
    stack: list[Atom] = [atom_list[0]]
    while stack:
        current = stack.pop()
        for bond in current.bonds:
            neighbor = bond.other(current)
            if neighbor in members and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited == members


def _members_have_multiple_bond(atom_list: list[Atom]) -> bool:
    """成员两两之间是否存在 order > 1 的显式键。"""
    members = set(atom_list)
    for atom in atom_list:
        for bond in atom.bonds:
            if bond.order > 1 and bond.other(atom) in members:
                return True
    return False


def _check_containers(molecule: Molecule) -> None:
    """容器一致性：四表双向一致、无重复、belong 正确。"""
    atom_ids: set[Atom] = set()
    for atom in molecule.atoms:
        if atom in atom_ids:
            raise ValueError("molecule.atoms 中存在重复原子")
        atom_ids.add(atom)
        if atom.belong is not molecule:
            raise ValueError("原子 belong 与所属分子不一致")
        seen: set[Bond] = set()
        for bond in atom.bonds:
            if bond in seen:
                raise ValueError("原子 bonds 列表中存在重复键")
            seen.add(bond)
            if bond not in molecule.bonds:
                raise ValueError("原子引用的键不在 molecule.bonds 中")
    bond_ids: set[Bond] = set()
    for bond in molecule.bonds:
        if bond in bond_ids:
            raise ValueError("molecule.bonds 中存在重复键")
        bond_ids.add(bond)
        atom1, atom2 = bond.atoms
        if atom1 not in atom_ids or atom2 not in atom_ids:
            raise ValueError("键端点不属于分子原子集合")
        if atom1.belong is not molecule or atom2.belong is not molecule:
            raise ValueError("键端点不属于同一分子")
        if bond not in atom1.bonds or bond not in atom2.bonds:
            raise ValueError("molecule.bonds 中的键未被两端原子引用")
    for pi in molecule.pi_systems:
        for atom in pi.atoms:
            if atom not in atom_ids:
                raise ValueError("π 体系成员不属于分子原子集合")


def _check_pi_system(pi: PiSystem) -> None:
    """π 体系完整性：成员数量、重复、连通、多重键、单价元素。"""
    if len(pi.atoms) < 2:
        raise ValueError("π 体系至少需要两个原子")
    if len(set(pi.atoms)) != len(pi.atoms):
        raise ValueError("π 体系成员不能重复")
    for atom in pi.atoms:
        if CHEMISTRY_BOND_DICT[atom.name] < 2:
            raise ValueError(f"单价元素 {atom.name} 不能参与 π 体系")
    if not _pi_members_connected(pi.atoms):
        raise ValueError("π 体系成员必须通过键彼此连通")
    if _members_have_multiple_bond(pi.atoms):
        raise ValueError("π 体系成员之间不能存在显式多重键")


def add_pi_bond(atom_list: list[Atom], dbe: int | None = None) -> PiSystem:
    """为一组原子建立离域 π 体系。

    校验顺序：成员数 ≥ 2 → 同分子 → 单价元素拒绝 → 无重复成员 →
    成员间无显式多重键 → 每原子最多参与一个 π 体系 → 价键容量。
    成员连通性在 Molecule.validate()（指纹计算前）统一校验，
    以允许"先建 π 体系、后补完 σ 骨架"的增量构建。
    """
    if len(atom_list) < 2:
        raise ValueError("π 体系至少需要两个原子")
    molecule = atom_list[0].belong
    if any(atom.belong is not molecule for atom in atom_list):
        raise ValueError("π 体系的所有原子必须属于同一分子")
    for atom in atom_list:
        if CHEMISTRY_BOND_DICT[atom.name] < 2:
            raise ValueError(f"单价元素 {atom.name} 不能参与 π 体系")
    if len(set(atom_list)) != len(atom_list):
        raise ValueError("π 体系成员不能重复")
    if _members_have_multiple_bond(atom_list):
        raise ValueError("π 体系成员之间已存在显式多重键")
    for atom in atom_list:
        if any(atom in pi.atoms for pi in molecule.pi_systems):
            raise ValueError(f"{atom.name} 原子已参与其他 π 体系")
    for atom in atom_list:
        _check_valence(atom, 1)
    pi = PiSystem(atom_list, dbe=dbe)
    molecule.pi_systems.append(pi)
    return pi


def break_pi_bond(pi_system: PiSystem) -> None:
    """移除一个 π 体系。"""
    molecule = pi_system.atoms[0].belong
    molecule.pi_systems.remove(pi_system)


def add_active_h(target_atom: Atom) -> ActiveH:
    """在 target_atom 的空价键槽位上挂一个活性氢（ActiveH 节点）。"""
    if target_atom.used_valence >= CHEMISTRY_BOND_DICT[target_atom.name]:
        raise ValueError(f"{target_atom.name} 原子没有可用的价键槽位")
    hydrogen = ActiveH(target_atom.belong)
    add_bond(target_atom, hydrogen)
    return hydrogen


def del_atom(atom: Atom) -> None:
    """删除原子：清理其所有键与 π 体系参与，并从分子中移除。"""
    molecule = atom.belong
    for bond in list(atom.bonds):
        for endpoint in bond.atoms:
            endpoint.bonds.remove(bond)
        molecule.bonds.remove(bond)
    atom.bonds.clear()
    for pi in list(molecule.pi_systems):
        if atom in pi.atoms:
            pi.atoms.remove(atom)
            if len(pi.atoms) < 2:
                molecule.pi_systems.remove(pi)
    molecule.atoms.remove(atom)


def connect(target_atom_list: list[Atom], is_cyclization: bool = False) -> None:
    """将一串原子用单键顺序相连；is_cyclization 为真时首尾相连成环（至少 3 个原子）。"""
    if len(target_atom_list) < 2:
        raise ValueError("至少需要两个原子")
    if is_cyclization and len(target_atom_list) < 3:
        raise ValueError("至少需要 3 个原子才能成环")
    for index in range(len(target_atom_list) - 1):
        add_bond(target_atom_list[index], target_atom_list[index + 1])
    if is_cyclization:
        add_bond(target_atom_list[0], target_atom_list[-1])


# ------- 结构指纹（Weisfeiler-Lehman） -------

def _infer_pi_dbe(atoms: list[Atom]) -> int:
    """根据 π 体系组成推断不饱和度贡献：全碳芳香体系 ≈ 原子数/2，硝基型 ≈ 1。"""
    element_counts: dict[str, int] = {}
    for atom in atoms:
        element_counts[atom.name] = element_counts.get(atom.name, 0) + 1
    if len(element_counts) == 1 and 'c' in element_counts:
        return len(atoms) // 2
    if element_counts.get('n') == 1 and element_counts.get('o') == 2 and len(atoms) == 3:
        return 1
    return 1


def _initial_label(atom: Atom) -> Label:
    # 类型即标记：ActiveH 子类与普通显式氢、隐氢区分开
    return (atom.name, atom.charge, isinstance(atom, ActiveH))


def _refine_label(atom: Atom, labels: dict[Atom, AtomLabel], molecule: Molecule) -> str:
    neighbor_info = tuple(sorted(
        (labels[bond.other(atom)], bond.order) for bond in atom.bonds
    ))
    pi_info: list[tuple[int, tuple[AtomLabel, ...]]] = []
    for pi in molecule.pi_systems:
        if atom in pi.atoms:
            others = tuple(sorted(labels[other] for other in pi.atoms if other is not atom))
            pi_info.append((pi.dbe, others))
    return _digest((_initial_label(atom), neighbor_info, tuple(sorted(pi_info))))


def _digest(value: object) -> str:
    return hashlib.md5(repr(value).encode('utf-8')).hexdigest()


def _wl_labels(molecule: Molecule) -> dict[Atom, AtomLabel]:
    """WL 迭代细化：返回每个原子的最终标签（颜色），供指纹与同构剪枝使用。"""
    labels: dict[Atom, AtomLabel] = {atom: _initial_label(atom) for atom in molecule.atoms}
    for _ in range(len(molecule.atoms)):
        new_labels: dict[Atom, AtomLabel] = {
            atom: _refine_label(atom, labels, molecule) for atom in molecule.atoms
        }
        if all(new_labels[atom] == labels[atom] for atom in molecule.atoms):
            labels = new_labels
            break
        labels = new_labels
    return labels


def _fingerprint(molecule: Molecule) -> list[str]:
    """返回排序后的稳定指纹列表：与建键顺序无关，作相等判断的预筛输入。"""
    if not molecule.atoms:
        return []
    molecule.validate()
    labels = _wl_labels(molecule)
    header = _digest((
        len(molecule.atoms), len(molecule.bonds),
        molecule.unsaturation, len(molecule.pi_systems),
    ))
    return [header] + sorted(_digest(labels[atom]) for atom in molecule.atoms)


def _build_adjacency(molecule: Molecule) -> dict[Atom, dict[Atom, tuple[int, ...]]]:
    """邻接表：端点对 -> 有序键级列表（防御直接构造造成的平行键）。"""
    neighbors: dict[Atom, dict[Atom, list[int]]] = {atom: {} for atom in molecule.atoms}
    for bond in molecule.bonds:
        atom1, atom2 = bond.atoms
        neighbors[atom1].setdefault(atom2, []).append(bond.order)
        neighbors[atom2].setdefault(atom1, []).append(bond.order)
    return {
        atom: {other: tuple(sorted(orders)) for other, orders in table.items()}
        for atom, table in neighbors.items()
    }


def _pi_index(molecule: Molecule) -> dict[Atom, tuple[int, int]]:
    """原子 -> (所属 π 体系序号, dbe)；未参与 π 体系的原子不在结果中。

    前置条件：结构已通过 validate()，每原子至多参与一个 π 体系。
    """
    atom_pi: dict[Atom, tuple[int, int]] = {}
    for index, pi in enumerate(molecule.pi_systems):
        for atom in pi.atoms:
            atom_pi[atom] = (index, pi.dbe)
    return atom_pi


def _are_isomorphic(m1: Molecule, m2: Molecule) -> bool:
    """精确同构判断：WL 颜色剪枝 + VF2 风格回溯确认。

    保持顶点标签（元素 / 形式电荷 / 是否 ActiveH）、键级、π 体系
    （dbe 与成员集合）三者一致；供 __eq__ 在 WL 预筛通过后做精确确认。
    """
    m1.validate()
    m2.validate()
    if not m1.atoms:
        return not m2.atoms
    if (len(m1.atoms), len(m1.bonds), len(m1.pi_systems)) != (
            len(m2.atoms), len(m2.bonds), len(m2.pi_systems)):
        return False

    colors1 = _wl_labels(m1)
    colors2 = _wl_labels(m2)
    if sorted(colors1.values()) != sorted(colors2.values()):
        return False

    adj1 = _build_adjacency(m1)
    adj2 = _build_adjacency(m2)
    atom_pi1 = _pi_index(m1)
    atom_pi2 = _pi_index(m2)

    mapping: dict[Atom, Atom] = {}
    matched2: set[Atom] = set()
    sys_map: dict[int, int] = {}      # m1 体系序号 -> m2 体系序号
    sys_map_rev: dict[int, int] = {}  # m2 体系序号 -> m1 体系序号

    def feasible(v1: Atom, w: Atom) -> bool:
        """候选 w 是否可与 v1 配对（基于已匹配映射）。"""
        if _initial_label(v1) != _initial_label(w):
            return False
        if colors1[v1] != colors2[w]:
            return False
        p1 = atom_pi1.get(v1)
        p2 = atom_pi2.get(w)
        if p1 is None:
            if p2 is not None:
                return False
        else:
            if p2 is None or p1[1] != p2[1]:
                return False
            if p1[0] in sys_map:
                if sys_map[p1[0]] != p2[0]:
                    return False
            elif p2[0] in sys_map_rev:
                return False
        for u1, u2 in mapping.items():
            if adj1[v1].get(u1, ()) != adj2[w].get(u2, ()):
                return False
        return True

    def backtrack() -> bool:
        if len(mapping) == len(m1.atoms):
            return True
        # fail-first：每次选候选集最小的未匹配顶点
        best_v1: Atom | None = None
        best_candidates: list[Atom] = []
        for v1 in m1.atoms:
            if v1 in mapping:
                continue
            candidates = [w for w in m2.atoms if w not in matched2 and feasible(v1, w)]
            if not candidates:
                return False
            if best_v1 is None or len(candidates) < len(best_candidates):
                best_v1 = v1
                best_candidates = candidates
                if len(candidates) == 1:
                    break
        assert best_v1 is not None
        for w in best_candidates:
            p1 = atom_pi1.get(best_v1)
            p2 = atom_pi2.get(w)
            added_pair: tuple[int, int] | None = None
            if p1 is not None and p1[0] not in sys_map:
                assert p2 is not None  # feasible() 已保证：p1 非 None 时 p2 必非 None
                added_pair = (p1[0], p2[0])
                sys_map[added_pair[0]] = added_pair[1]
                sys_map_rev[added_pair[1]] = added_pair[0]
            mapping[best_v1] = w
            matched2.add(w)
            if backtrack():
                return True
            matched2.remove(w)
            del mapping[best_v1]
            if added_pair is not None:
                del sys_map[added_pair[0]]
                del sys_map_rev[added_pair[1]]
        return False

    return backtrack()


def _display_formula(formula: dict[str, int]) -> str:
    parts: list[str] = []
    for element, num in formula.items():
        if num == 0:
            continue
        display = element.capitalize()
        parts.append(display + (str(num) if num != 1 else ''))
    return ''.join(parts)
