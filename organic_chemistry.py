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
- 指纹快照契约：结构指纹是快照，任何编辑操作（建/断键、增删 π 体系、
  增删原子等）会自动失效快照；失效后读取 feature 或做 == 判等会自动
  校验并重算（惰性重建），也可显式调用 Molecule.update() 提前校验。
  直接修改 bond.order / atom.charge / pi.dbe 等标量属性、以及直接构造
  Bond / PiSystem 均属非法操作，一律抛 ValueError；所有结构变更必须
  通过编辑 API（add_bond / break_bond / add_pi_system / remove_pi_system /
  add_active_h / del_atom）进行。
- 表示公约：显式多重键仅表示局域键；同一 π 体系成员之间只允许单键，
  离域体系一律用 add_pi_system 创建 PiSystem 表示，不得用多重键代替。
  违反该公约视为无效结构，在编辑、validate() 与派生性质计算时抛 ValueError。
- 苯环约定：凯库勒式（环内交替单双键）与鲍林式（六根单键 + PiSystem）
  两种画法完全等价；构建时检测到交替六元碳环会自动归一化为单键 + PiSystem。
- 硝基型 [N,O,O] π 体系中的 N 允许 4 个价键槽位（等效 N⁺ 的 4 键表示）；
  硝基一律用 3 个单键 + PiSystem 表示，显式 N=O 双键画法仍不允许。
"""

from __future__ import annotations

import hashlib
from typing import Literal, Sequence, TypeAlias, cast

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

class _FingerprintSnapshot:
    """update() 后生效的指纹快照：feature 列表 + WL 原子标签。"""

    def __init__(self, feature: list[str], labels: dict[Atom, AtomLabel]) -> None:
        self.feature: list[str] = feature
        self.labels: dict[Atom, AtomLabel] = labels


class Molecule:
    """分子：原子、键、π 体系的容器，以及所有派生性质的现算入口。"""

    def __init__(self, name: str | None = None) -> None:
        self.name: str | None = name
        self.atoms: list[Atom] = []
        self.bonds: list[Bond] = []
        self.pi_systems: list[PiSystem] = []
        self._snapshot: _FingerprintSnapshot | None = None  # update() 后生效的指纹快照

    def _invalidate(self) -> None:
        """编辑后失效指纹快照；下次读取 feature / == 判等时自动重算。"""
        self._snapshot = None

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
        return _unsaturation_value(self)

    def validate(self) -> None:
        """校验结构不变量；无效结构抛 ValueError。

        检查范围：
        - 容器一致性：molecule.bonds 与各 atom.bonds 双向一致、
          无重复、atom.belong 正确、键与 π 成员属于分子；
        - 键自身：1 <= order <= MAX_BOND_ORDER，无自环；
        - 价键：每个原子的 used_valence 不超过其价键数；
        - π 体系：成员 >= 2、无重复、连通、成员间无显式多重键、
          每原子至多参与一个 π 体系、单价元素不参与。
        update()、读取 feature / 判等与 unsaturation 计算前会自动调用。
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

        编辑操作会自动失效快照；读取 feature 或做 == 判等时会自动
        校验并重算（惰性重建）。本方法是可选的显式入口：需要提前
        校验（fail-fast）或主动刷新快照时调用；无效结构抛 ValueError。
        直接修改 bond.order / atom.charge / pi.dbe 等标量属性、
        以及直接构造 Bond / PiSystem 属非法操作，会抛 ValueError。
        """
        self.validate()
        self._snapshot = _build_snapshot(self)

    @property
    def feature(self) -> list[str]:
        """结构指纹快照（与建键顺序无关）；快照过期时自动校验并重算。

        返回防御性拷贝：外部修改返回值不影响内部缓存。
        无效结构在自动重算时抛 ValueError。
        """
        if self._snapshot is None:
            self.update()
        assert self._snapshot is not None
        return list(self._snapshot.feature)

    @property
    def equivalent_hydrogen_groups(self) -> list[int]:
        """等位氢分组：每种化学环境的氢原子个数列表。

        等位氢 = 同一原子上的全部氢（隐氢 / 显式 H / ActiveH 合并），
        以及可被自同构相互映对的等价原子上的氢；按连通分量分别计算，
        避免把不同分子的氢误并为一组。结果按个数降序排列（同个数按
        分子内原子首次出现顺序），例如乙醇 → [3, 2, 1]。
        计算前先校验结构（validate），无效结构抛 ValueError。
        """
        self.validate()
        if not self.atoms:
            return []
        colors: dict[Atom, AtomLabel]
        if self._snapshot is not None:
            colors = self._snapshot.labels
        else:
            colors = _wl_labels(self)
        orbit_id = _automorphism_orbits(self, colors)
        counts: dict[int, int] = {}
        first_index: dict[int, int] = {}
        for index, atom in enumerate(self.atoms):
            if atom.name != 'h':
                total = _total_h(atom)
                if total <= 0:
                    continue
                gid = orbit_id[atom]
            else:
                # 显式 H 节点：挂在非氢原子上时由宿主原子统一计数；
                # 否则（如 H2、孤立 H）按自身轨道自成一组。
                if any(bond.other(atom).name != 'h' for bond in atom.bonds):
                    continue
                gid = orbit_id[atom]
                total = 1
            counts[gid] = counts.get(gid, 0) + total
            first_index.setdefault(gid, index)
        ordered = sorted(counts, key=lambda gid: (-counts[gid], first_index[gid]))
        return [counts[gid] for gid in ordered]

    # ---- 内置方法 ----

    def __eq__(self, other: object) -> bool:
        """精确结构判等：WL 指纹快照预筛 + 回溯同构确认。

        双方快照过期时自动校验并重算；无效结构抛 ValueError。
        """
        if not isinstance(other, Molecule):
            return False
        if self.feature != other.feature:
            return False
        assert self._snapshot is not None and other._snapshot is not None
        return _are_isomorphic(
            self, other,
            colors1=self._snapshot.labels,
            colors2=other._snapshot.labels,
        )

    # 注意：Molecule 定义了 __eq__ 且结构可变，因此不定义 __hash__，
    # Python 会自动令其实例不可哈希；需要作集合/字典键时用 tuple(molecule.feature)。
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Molecule({_display_formula(self.formula)})"


class Atom:
    """原子：只记录身份与连接关系，特征全部派生。"""

    _charge: int  # 形式电荷（预留：不影响价态/分子式，当前仅参与判等指纹）

    def __init__(self, name: str, molecule: Molecule) -> None:
        name = name.lower()
        if name not in CHEMISTRY_BOND_DICT:
            raise ValueError(f"{name} 不是可成键元素：{sorted(CHEMISTRY_BOND_DICT)}")
        self.name: ElementName = cast(ElementName, name)
        self.bonds: list[Bond] = []   # 参与的所有键（Bond 对象）
        self._charge: int = 0         # 形式电荷（预留：不影响价态/分子式，当前仅参与判等指纹）
        self.belong: Molecule = molecule
        molecule.atoms.append(self)
        molecule._invalidate()

    @property
    def charge(self) -> int:
        """形式电荷（预留：不影响价态/分子式，当前仅参与判等指纹）。只读。"""
        return self._charge

    @charge.setter
    def charge(self, value: int) -> None:
        raise ValueError("形式电荷暂不处理：禁止直接修改 atom.charge")

    @property
    def used_valence(self) -> int:
        """已占用的价键数 = 键级之和 + π 体系槽位数。"""
        used = sum(bond.order for bond in self.bonds)
        used += sum(1 for pi in self.belong.pi_systems if self in pi.atoms)
        return used

    @property
    def implicit_h(self) -> int:
        """隐氢数 = 价键容量 - 已占用价键数；单价元素不计隐氢（空槽位为自由基）。"""
        if CHEMISTRY_BOND_DICT[self.name] < 2:
            return 0
        return max(0, _valence_limit(self) - self.used_valence)

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
    """键：一等对象，两个端点 + 键级 + 预留的立体化学字段。

    禁止直接构造：必须通过 add_bond() 创建；order/stereo/aromatic
    为只读属性，直接修改会抛 ValueError。
    """

    _atoms: tuple[Atom, Atom]
    _order: int
    _stereo: str | None
    _aromatic: bool

    def __init__(
        self,
        atom1: Atom,
        atom2: Atom,
        order: int = 1,
        stereo: str | None = None,
        aromatic: bool = False,
    ) -> None:
        raise ValueError("不允许直接构造 Bond；请使用 add_bond()")

    @property
    def atoms(self) -> tuple[Atom, Atom]:
        """键的两个端点。只读。"""
        return self._atoms

    @atoms.setter
    def atoms(self, value: tuple[Atom, Atom]) -> None:
        raise ValueError("键端点不能直接修改；请使用 add_bond()/break_bond()")

    @property
    def order(self) -> int:
        """键级。只读，只能通过 add_bond()/break_bond() 修改。"""
        return self._order

    @order.setter
    def order(self, value: int) -> None:
        raise ValueError("键级只能通过 add_bond()/break_bond() 修改")

    @property
    def stereo(self) -> str | None:
        """预留：顺反异构/立体构型（当前不参与任何计算）。只读。"""
        return self._stereo

    @stereo.setter
    def stereo(self, value: str | None) -> None:
        raise ValueError("stereo 为预留字段，禁止直接修改")

    @property
    def aromatic(self) -> bool:
        """预留：芳香键标记（当前不参与任何计算；芳香性由 PiSystem 表示）。只读。"""
        return self._aromatic

    @aromatic.setter
    def aromatic(self, value: bool) -> None:
        raise ValueError("aromatic 为预留字段，禁止直接修改")

    def other(self, atom: Atom) -> Atom:
        """返回键的另一端原子。"""
        atom1, atom2 = self._atoms
        return atom2 if atom is atom1 else atom1

    def __repr__(self) -> str:
        atom1, atom2 = self._atoms
        return f"<Bond {atom1.name}-{atom2.name} order={self._order}>"


def _make_bond(
    atom1: Atom,
    atom2: Atom,
    order: int = 1,
    stereo: str | None = None,
    aromatic: bool = False,
) -> Bond:
    """内部构造器：add_bond() 专用；执行与旧 Bond.__init__ 相同的防御校验。

    不校验价键容量（由 add_bond() 负责）；测试可用它模拟损坏结构。
    """
    if atom1 is atom2:
        raise ValueError("原子不能与自身成键")
    if atom1.belong is not atom2.belong:
        raise ValueError("不能连接不同分子的原子")
    if not 1 <= order <= MAX_BOND_ORDER:
        raise ValueError(f"键级必须为 1-{MAX_BOND_ORDER}")
    bond = object.__new__(Bond)
    bond._atoms = (atom1, atom2)
    bond._order = order
    bond._stereo = stereo
    bond._aromatic = aromatic
    atom1.bonds.append(bond)
    atom2.bonds.append(bond)
    atom1.belong.bonds.append(bond)
    atom1.belong._invalidate()
    return bond


class PiSystem:
    """离域 π 体系：一组原子 + 不饱和度贡献（dbe）。

    表示公约：离域体系一律用 PiSystem 表示，成员之间只允许单键；
    显式多重键仅表示局域键，不得与 π 体系混用于同一对原子。
    禁止直接构造：必须通过 add_pi_system() 创建；dbe/aromatic 只读。
    """

    _atoms: list[Atom]
    _dbe: int
    _aromatic: bool

    def __init__(
        self,
        atoms: list[Atom],
        dbe: int | None = None,
        aromatic: bool = False,
    ) -> None:
        raise ValueError("不允许直接构造 PiSystem；请使用 add_pi_system()")

    @property
    def atoms(self) -> list[Atom]:
        """π 体系成员。只读（列表本身仍可原地操作）。"""
        return self._atoms

    @atoms.setter
    def atoms(self, value: list[Atom]) -> None:
        raise ValueError("π 体系成员不能直接重新绑定；请使用 add_pi_system()/remove_pi_system()")

    @property
    def dbe(self) -> int:
        """不饱和度贡献。只读，创建时由 add_pi_system() 指定。"""
        return self._dbe

    @dbe.setter
    def dbe(self, value: int) -> None:
        raise ValueError("dbe 只能在 add_pi_system() 创建时指定")

    @property
    def aromatic(self) -> bool:
        """预留：芳香性标记（当前不参与任何计算）。只读。"""
        return self._aromatic

    @aromatic.setter
    def aromatic(self, value: bool) -> None:
        raise ValueError("aromatic 为预留字段，禁止直接修改")

    def __repr__(self) -> str:
        names = ','.join(atom.name for atom in self._atoms)
        return f"<PiSystem({names}) dbe={self._dbe}>"


def _make_pi_system(
    atoms: list[Atom],
    dbe: int | None = None,
    aromatic: bool = False,
) -> PiSystem:
    """内部构造器：add_pi_system() 专用；保留成员数防御校验。"""
    if len(atoms) < 2:
        raise ValueError("π 体系至少需要两个原子")
    pi = object.__new__(PiSystem)
    pi._atoms = list(atoms)
    pi._dbe = _infer_pi_dbe(pi._atoms) if dbe is None else dbe
    pi._aromatic = aromatic
    return pi


# ------- 构建与编辑操作 -------

def _find_bond(atom1: Atom, atom2: Atom) -> Bond | None:
    for bond in atom1.bonds:
        if bond.other(atom1) is atom2:
            return bond
    return None


def _is_nitro_pi(atoms: list[Atom]) -> bool:
    """是否为硝基型 π 体系：3 个成员且恰好 1 个 n、2 个 o（按组成判定）。"""
    if len(atoms) != 3:
        return False
    counts: dict[str, int] = {}
    for atom in atoms:
        counts[atom.name] = counts.get(atom.name, 0) + 1
    return counts.get('n') == 1 and counts.get('o') == 2


def _valence_limit(atom: Atom) -> int:
    """原子的价键容量：基础价；硝基型 π 体系中的 n 额外 +1（等效 N⁺ 的 4 键）。"""
    limit = CHEMISTRY_BOND_DICT[atom.name]
    if atom.name == 'n':
        for pi in atom.belong.pi_systems:
            if atom in pi.atoms and _is_nitro_pi(pi.atoms):
                limit += 1
                break
    return limit


def _check_valence(atom: Atom, delta: int, extra: int = 0) -> None:
    used = atom.used_valence + delta
    limit = _valence_limit(atom) + extra
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

    表示公约：同一 π 体系成员之间只允许单键；离域体系请用 add_pi_system，
    不要用显式多重键代替。苯环例外：凯库勒式（环内交替单双键）成形时
    自动归一化为单键 + PiSystem，与隐式画法等价。
    """
    bond = _find_bond(atom1, atom2)
    if bond is not None:
        new_order = bond.order + order
        if new_order > MAX_BOND_ORDER:
            raise ValueError(f"键级不能超过 {MAX_BOND_ORDER}")
        _check_multiple_bond_within_pi(atom1, atom2, new_order)
        _check_valence(atom1, order)
        _check_valence(atom2, order)
        bond._order = new_order
        atom1.belong._invalidate()
        _normalize_benzene_rings(atom1.belong)
        return bond
    _check_multiple_bond_within_pi(atom1, atom2, order)
    _check_valence(atom1, order)
    _check_valence(atom2, order)
    result = _make_bond(atom1, atom2, order)
    _normalize_benzene_rings(atom1.belong)
    return result


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
        bond._order -= order
        atom1.belong._invalidate()
        return
    molecule = atom1.belong
    for atom in bond.atoms:
        atom.bonds.remove(bond)
    molecule.bonds.remove(bond)
    molecule._invalidate()


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
            if atom not in bond.atoms:
                raise ValueError("原子 bonds 列表中存在不以该原子为端点的键")
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


def add_pi_system(atom_list: list[Atom], dbe: int | None = None) -> PiSystem:
    """为一组原子建立离域 π 体系。

    校验顺序：成员数 ≥ 2 → 同分子 → 单价元素拒绝 → 无重复成员 →
    成员间无显式多重键 → 每原子最多参与一个 π 体系 → 价键容量。
    价键容量按 _valence_limit 计算：硝基型 [N,O,O] 中的 N 允许 4 个槽位
    （等效 N⁺），其余场合不豁免；硝基请用 3 个单键 + 本函数表示，
    显式 N=O 双键画法仍不允许。
    苯环特判：成员恰好为交替六元碳环（凯库勒式）时自动降环内双键并继续，
    显式画法与隐式画法等价。
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
        # 苯环特判：恰为一个凯库勒式交替环时，先降环内双键再按隐式画法继续
        if _is_benzene_ring_atom_list(atom_list):
            _lower_ring_double_bonds(atom_list)
        else:
            raise ValueError("π 体系成员之间已存在显式多重键")
    for atom in atom_list:
        if any(atom in pi.atoms for pi in molecule.pi_systems):
            raise ValueError(f"{atom.name} 原子已参与其他 π 体系")
    nitro_extra = 1 if _is_nitro_pi(atom_list) else 0
    for atom in atom_list:
        _check_valence(atom, 1, extra=nitro_extra if atom.name == 'n' else 0)
    pi = _make_pi_system(atom_list, dbe=dbe)
    molecule.pi_systems.append(pi)
    molecule._invalidate()
    return pi


def remove_pi_system(pi_system: PiSystem) -> None:
    """移除一个 π 体系。"""
    molecule = pi_system.atoms[0].belong
    molecule.pi_systems.remove(pi_system)
    molecule._invalidate()


def add_active_h(target_atom: Atom) -> ActiveH:
    """在 target_atom 的空价键槽位上挂一个活性氢（ActiveH 节点）。"""
    if target_atom.used_valence >= CHEMISTRY_BOND_DICT[target_atom.name]:
        raise ValueError(f"{target_atom.name} 原子没有可用的价键槽位")
    hydrogen = ActiveH(target_atom.belong)
    add_bond(target_atom, hydrogen)
    return hydrogen


def del_atom(atom: Atom) -> None:
    """删除原子：清理其所有键与 π 体系参与，并从分子中移除。

    π 体系处理：只把该原子从每个所属 π 体系中摘除；摘除后成员数不足 2 个时才
    移除整个 π 体系，否则保留剩余成员（剩余成员空出的价键槽位按隐式氢补足）。
    所以单独调用本函数删苯环/硝基成员时，π 体系本身不会被删掉。
    Web 编辑器（oc_web.edit_molecule() 的 del_atom 分支）删除 π 体系成员时，
    就是删掉整个 π 体系：先对该原子所在的每个 π 体系调用 remove_pi_system()，
    再调用本函数删除该原子；π 体系的其余成员原子保留在分子中，不被连带删除。
    """
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
    molecule._invalidate()


def connect(target_atom_list: list[Atom], is_cyclization: bool = False) -> None:
    """将一串原子用单键顺序相连；is_cyclization 为真时首尾相连成环（至少 3 个原子）。

    原子性：先对整个连接序列做累积预校验（同分子、非自环、同一原子对不重复、
    键级上限、π 体系公约、价键容量），全部通过后再统一建键/升键级；
    任一步失败都不会留下部分键或部分升键。
    序列中的原子应互异（同一无序原子对至多出现一次）；双键请用 add_bond 升级，
    不要通过重复连接同一对原子表达。
    """
    if len(target_atom_list) < 2:
        raise ValueError("至少需要两个原子")
    if is_cyclization and len(target_atom_list) < 3:
        raise ValueError("至少需要 3 个原子才能成环")
    pairs: list[tuple[Atom, Atom]] = [
        (target_atom_list[index], target_atom_list[index + 1])
        for index in range(len(target_atom_list) - 1)
    ]
    if is_cyclization:
        pairs.append((target_atom_list[0], target_atom_list[-1]))

    # 预校验：模拟整段连接对每个原子价键与已有键键级的累积占用
    pending_valence: dict[Atom, int] = {}
    pending_upgrade: dict[Bond, int] = {}
    seen_pairs: set[frozenset[Atom]] = set()
    for atom1, atom2 in pairs:
        pair = frozenset((atom1, atom2))
        if pair in seen_pairs:
            raise ValueError("连接序列中同一对原子不能重复出现")
        seen_pairs.add(pair)
        if atom1.belong is not atom2.belong:
            raise ValueError("不能连接不同分子的原子")
        if atom1 is atom2:
            raise ValueError("原子不能与自身成键")
        bond = _find_bond(atom1, atom2)
        if bond is not None:
            upgrade = pending_upgrade.get(bond, 0) + 1
            new_order = bond.order + upgrade
            if new_order > MAX_BOND_ORDER:
                raise ValueError(f"键级不能超过 {MAX_BOND_ORDER}")
            _check_multiple_bond_within_pi(atom1, atom2, new_order)
            _check_valence(atom1, pending_valence.get(atom1, 0) + 1)
            _check_valence(atom2, pending_valence.get(atom2, 0) + 1)
            pending_upgrade[bond] = upgrade
        else:
            _check_multiple_bond_within_pi(atom1, atom2, 1)
            _check_valence(atom1, pending_valence.get(atom1, 0) + 1)
            _check_valence(atom2, pending_valence.get(atom2, 0) + 1)
        pending_valence[atom1] = pending_valence.get(atom1, 0) + 1
        pending_valence[atom2] = pending_valence.get(atom2, 0) + 1

    # 应用：预校验已全部通过，按序建键/升键级
    for atom1, atom2 in pairs:
        add_bond(atom1, atom2)


def ensure_single_component(molecule: Molecule, role: str = "分子") -> None:
    """确认 molecule 只有一个连通分量，否则抛 ValueError。

    保存、同分异构体分析与合成路线分析都以"一个分子"为前提：出现多个
    互不成键的片段时，分子式、基团与合成分析都没有意义，直接拒绝。
    """
    count = molecule.component_count
    if count == 1:
        return
    if count == 0:
        raise ValueError(f"{role}不能为空（没有任何原子）")
    raise ValueError(
        f"{role}必须只有一个连通分量，当前有 {count} 个互不成键的片段"
    )


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


def _find_benzene_rings(molecule: Molecule) -> list[tuple[Atom, ...]]:
    """枚举凯库勒式苯环：6 个碳构成的交替单双键简单环。

    以环中 id 最小的原子为起点做 DFS，枚举长度 6 的简单环（每环恰好枚举一次），
    返回有序原子元组；环边为相邻原子对加首尾原子对。
    """
    rings: list[tuple[Atom, ...]] = []
    seen: set[frozenset[Atom]] = set()
    for start in molecule.atoms:
        if start.name != 'c':
            continue
        path: list[Atom] = [start]
        start_id = id(start)

        def visit() -> None:
            if len(path) == 6:
                last = path[-1]
                closing = _find_bond(last, start)
                if closing is None:
                    return
                orders: list[int] = []
                for i in range(5):
                    bond = _find_bond(path[i], path[i + 1])
                    assert bond is not None
                    orders.append(bond.order)
                orders.append(closing.order)
                # 交替判定：偶数位键级相同、奇数位键级相同，且两者不同
                if orders[0] == orders[1] or any(
                        orders[i] != orders[i % 2] for i in range(6)):
                    return
                ring_set = frozenset(path)
                if ring_set in seen:
                    return
                # 无弦校验：环内每个原子恰好 2 条环内键
                for atom in path:
                    ring_neighbors = sum(
                        1 for bond in atom.bonds if bond.other(atom) in ring_set
                    )
                    if ring_neighbors != 2:
                        return
                seen.add(ring_set)
                rings.append(tuple(path))
                return
            current = path[-1]
            for bond in current.bonds:
                neighbor = bond.other(current)
                if neighbor is start and len(path) > 1:
                    continue
                if neighbor.name != 'c' or id(neighbor) <= start_id:
                    continue
                if neighbor in path:
                    continue
                path.append(neighbor)
                visit()
                path.pop()

        visit()
    return rings


def _lower_ring_double_bonds(ring: list[Atom] | tuple[Atom, ...]) -> None:
    """把环内双键降为单键（凯库勒式 → 单键骨架）。"""
    for i in range(len(ring)):
        bond = _find_bond(ring[i], ring[(i + 1) % len(ring)])
        assert bond is not None
        if bond.order > 1:
            bond._order = 1


def _is_benzene_ring_atom_list(atom_list: list[Atom]) -> bool:
    """atom_list 是否恰好构成一个凯库勒式苯环（交替六元碳环）。"""
    if len(atom_list) != 6:
        return False
    target = frozenset(atom_list)
    molecule = atom_list[0].belong
    return any(frozenset(ring) == target for ring in _find_benzene_rings(molecule))


def _normalize_benzene_rings(molecule: Molecule) -> None:
    """构建时归一化：把凯库勒式苯环转成隐式表示（单键骨架 + PiSystem）。

    共享原子的交替环（稠环）合并为一个 π 体系，dbe 沿用 _infer_pi_dbe；
    与已有真实 π 体系共享原子的环（病态状态）跳过，不自动转换。
    """
    rings = _find_benzene_rings(molecule)
    if not rings:
        return
    parent = list(range(len(rings)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(rings)):
        ring_i = set(rings[i])
        for j in range(i + 1, len(rings)):
            if ring_i & set(rings[j]):
                union(i, j)

    groups: dict[int, list[tuple[Atom, ...]]] = {}
    for index, ring in enumerate(rings):
        groups.setdefault(find(index), []).append(ring)

    changed = False
    for group in groups.values():
        atoms: list[Atom] = []
        seen_atoms: set[Atom] = set()
        edges: set[frozenset[Atom]] = set()
        for ring in group:
            for i in range(len(ring)):
                atom_a, atom_b = ring[i], ring[(i + 1) % len(ring)]
                edges.add(frozenset((atom_a, atom_b)))
                if atom_a not in seen_atoms:
                    seen_atoms.add(atom_a)
                    atoms.append(atom_a)
        # 病态状态：与真实 π 体系重叠时交由 validate 报错，不自动转换
        if any(any(atom in pi.atoms for pi in molecule.pi_systems) for atom in atoms):
            continue
        for edge in edges:
            atom_a, atom_b = tuple(edge)
            bond = _find_bond(atom_a, atom_b)
            if bond is not None and bond.order > 1:
                bond._order = 1
                changed = True
        molecule.pi_systems.append(_make_pi_system(atoms, dbe=_infer_pi_dbe(atoms)))
        changed = True
    if changed:
        molecule._invalidate()


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


def _unsaturation_value(molecule: Molecule) -> int:
    """不饱和度数值（前置条件：结构已通过 validate()）。"""
    bond_part = sum(bond.order - 1 for bond in molecule.bonds)
    return bond_part + molecule.ring_count + sum(pi.dbe for pi in molecule.pi_systems)


def _build_snapshot(molecule: Molecule) -> _FingerprintSnapshot:
    """构造指纹快照（前置条件：结构已通过 validate()）。"""
    if not molecule.atoms:
        return _FingerprintSnapshot([], {})
    labels = _wl_labels(molecule)
    header = _digest((
        len(molecule.atoms), len(molecule.bonds),
        _unsaturation_value(molecule), len(molecule.pi_systems),
    ))
    feature = [header] + sorted(_digest(labels[atom]) for atom in molecule.atoms)
    return _FingerprintSnapshot(feature, labels)


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


def _are_isomorphic(
    m1: Molecule,
    m2: Molecule,
    colors1: dict[Atom, AtomLabel] | None = None,
    colors2: dict[Atom, AtomLabel] | None = None,
    fixed: dict[Atom, Atom] | None = None,
) -> bool:
    """精确同构判断：WL 颜色剪枝 + VF2 风格回溯确认。

    保持顶点标签（元素 / 形式电荷 / 是否 ActiveH）、键级、π 体系
    （dbe 与成员集合）三者一致；供 __eq__ 在 WL 预筛通过后做精确确认。
    colors1/colors2 传入 update() 缓存的 WL 标签时可跳过 validate() 与
    WL 迭代（__eq__ 使用）；缺省时自行校验并计算。
    fixed 预置起点映射（m1 原子 -> m2 原子，如 {a: b}），用于确认是否
    存在把 a 映到 b 的（自）同构；固定项会同步初始化 π 体系映射。
    """
    if colors1 is None:
        m1.validate()
        colors1 = _wl_labels(m1)
    if colors2 is None:
        m2.validate()
        colors2 = _wl_labels(m2)
    if not m1.atoms:
        return not m2.atoms
    if (len(m1.atoms), len(m1.bonds), len(m1.pi_systems)) != (
            len(m2.atoms), len(m2.bonds), len(m2.pi_systems)):
        return False

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

    if fixed is not None:
        for v1, w in fixed.items():
            if _initial_label(v1) != _initial_label(w) or colors1[v1] != colors2[w]:
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
                else:
                    sys_map[p1[0]] = p2[0]
                    sys_map_rev[p2[0]] = p1[0]
            mapping[v1] = w
            matched2.add(w)

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


def _automorphism_orbits(
    molecule: Molecule,
    colors: dict[Atom, AtomLabel],
) -> dict[Atom, int]:
    """精确自同构轨道：返回每个原子所属轨道的编号。

    同轨道的原子可被某个自同构相互映对（即严格等价位置）。以 WL 颜色
    与连通分量剪枝：只在同一分量、同一颜色的原子间两两做“固定映射
    回溯确认”，因此 WL 标签可能误并的不等价位置会被正确拆开。
    前置条件：结构已通过 validate()，colors 与之匹配。
    """
    comp: dict[Atom, int] = {}
    next_comp = 0
    for atom in molecule.atoms:
        if atom in comp:
            continue
        comp[atom] = next_comp
        stack: list[Atom] = [atom]
        while stack:
            current = stack.pop()
            for bond in current.bonds:
                neighbor = bond.other(current)
                if neighbor not in comp:
                    comp[neighbor] = next_comp
                    stack.append(neighbor)
        next_comp += 1

    parent: dict[Atom, Atom] = {atom: atom for atom in molecule.atoms}

    def find(atom: Atom) -> Atom:
        while parent[atom] is not atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    def union(a: Atom, b: Atom) -> None:
        root_a, root_b = find(a), find(b)
        if root_a is not root_b:
            parent[root_a] = root_b

    groups: dict[tuple[int, AtomLabel], list[Atom]] = {}
    for atom in molecule.atoms:
        groups.setdefault((comp[atom], colors[atom]), []).append(atom)

    for group in groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if find(a) is find(b):
                    continue
                if _are_isomorphic(molecule, molecule, colors, colors, fixed={a: b}):
                    union(a, b)

    orbit_id: dict[Atom, int] = {}
    for atom in molecule.atoms:
        root = find(atom)
        if root not in orbit_id:
            orbit_id[root] = len(orbit_id)
        orbit_id[atom] = orbit_id[root]
    return orbit_id


# ------- 子结构匹配与分子克隆 -------


class PatternAtom:
    """子结构模式中的一个原子。

    element 为 None 时匹配任意元素；h 为精确总氢数，h_min 为最少总氢数（互斥）；
    pi 为 True/False 时要求目标原子属于/不属于某个 π 体系，None 表示不限；
    pi_group 非 None 时，同值的模式原子必须落入同一个目标 π 体系。
    """

    def __init__(
        self,
        element: str | None = None,
        *,
        h: int | None = None,
        h_min: int | None = None,
        pi: bool | None = None,
        pi_group: int | None = None,
    ) -> None:
        if element is not None:
            element = element.lower()
            if element not in CHEMISTRY_BOND_DICT:
                raise ValueError(
                    f"{element} 不是可匹配的元素：{sorted(CHEMISTRY_BOND_DICT)} 或 None"
                )
        if h is not None and h < 0:
            raise ValueError("h 必须 >= 0")
        if h_min is not None and h_min < 0:
            raise ValueError("h_min 必须 >= 0")
        if h is not None and h_min is not None:
            raise ValueError("h 与 h_min 不能同时设置")
        if pi_group is not None and pi_group < 0:
            raise ValueError("pi_group 必须 >= 0")
        if pi_group is not None and pi is False:
            raise ValueError("pi_group 非 None 时 pi 不能为 False")
        self.element: str | None = element
        self.h: int | None = h
        self.h_min: int | None = h_min
        self.pi: bool | None = pi
        self.pi_group: int | None = pi_group

    def __repr__(self) -> str:
        parts = [self.element if self.element is not None else '*']
        if self.h is not None:
            parts.append(f'H={self.h}')
        if self.h_min is not None:
            parts.append(f'H>={self.h_min}')
        if self.pi is True:
            parts.append('pi')
        elif self.pi is False:
            parts.append('no-pi')
        if self.pi_group is not None:
            parts.append(f'pg{self.pi_group}')
        return '<PatternAtom ' + ' '.join(parts) + '>'


def _check_pattern_order(order: int | tuple[int, ...] | None) -> None:
    """校验模式键级；非法抛 ValueError。"""
    if order is None:
        return
    if isinstance(order, int):
        if not 1 <= order <= MAX_BOND_ORDER:
            raise ValueError(f"键级必须为 1-{MAX_BOND_ORDER}")
        return
    if not isinstance(order, tuple):
        raise ValueError("order 必须为 int、tuple 或 None")
    if not order:
        raise ValueError("order 元组不能为空")
    for value in order:
        if not isinstance(value, int) or not 1 <= value <= MAX_BOND_ORDER:
            raise ValueError(f"键级必须为 1-{MAX_BOND_ORDER} 的整数或元组")


class PatternBond:
    """子结构模式中的一条键。

    order 为 None 时匹配任意键级（1-3）；为 int 时精确匹配；为元组时匹配其中任一键级。
    """

    def __init__(
        self,
        atom1: PatternAtom,
        atom2: PatternAtom,
        order: int | tuple[int, ...] | None = None,
    ) -> None:
        if atom1 is atom2:
            raise ValueError("模式键不能连接同一个原子（自环）")
        _check_pattern_order(order)
        self.atom1: PatternAtom = atom1
        self.atom2: PatternAtom = atom2
        self.order: int | tuple[int, ...] | None = order

    def __repr__(self) -> str:
        return f"<PatternBond {self.atom1}..{self.atom2} order={self.order}>"


class Pattern:
    """子结构模式：一组模式原子与模式键。

    用 add_atom()/add_bond() 构建，构建即校验；匹配时目标分子允许有模式之外的
    额外键（子结构语义，非诱导子图）。
    """

    def __init__(self) -> None:
        self.atoms: list[PatternAtom] = []
        self.bonds: list[PatternBond] = []

    def add_atom(
        self,
        element: str | None = None,
        *,
        h: int | None = None,
        h_min: int | None = None,
        pi: bool | None = None,
        pi_group: int | None = None,
    ) -> PatternAtom:
        atom = PatternAtom(element, h=h, h_min=h_min, pi=pi, pi_group=pi_group)
        self.atoms.append(atom)
        return atom

    def add_bond(
        self,
        atom1: PatternAtom,
        atom2: PatternAtom,
        order: int | tuple[int, ...] | None = None,
    ) -> PatternBond:
        if atom1 not in self.atoms or atom2 not in self.atoms:
            raise ValueError("模式键的两个端点必须属于该模式")
        pair = frozenset((atom1, atom2))
        for bond in self.bonds:
            if frozenset((bond.atom1, bond.atom2)) == pair:
                raise ValueError("同一对模式原子之间只能有一条模式键")
        bond = PatternBond(atom1, atom2, order=order)
        self.bonds.append(bond)
        return bond

    def __repr__(self) -> str:
        return f"<Pattern atoms={len(self.atoms)} bonds={len(self.bonds)}>"


class SubstructureMatch:
    """一次子结构匹配结果。

    atom_map: 模式原子 -> 目标原子；pi_map: pi_group -> 目标 π 体系；
    atoms: 按 pattern.atoms 顺序排列的目标原子。
    """

    def __init__(
        self,
        atom_map: dict[PatternAtom, Atom],
        pi_map: dict[int, PiSystem],
        pattern: Pattern,
    ) -> None:
        self.atom_map: dict[PatternAtom, Atom] = dict(atom_map)
        self.pi_map: dict[int, PiSystem] = dict(pi_map)
        self.atoms: tuple[Atom, ...] = tuple(atom_map[atom] for atom in pattern.atoms)

    def __repr__(self) -> str:
        return '<SubstructureMatch ' + ','.join(atom.name for atom in self.atoms) + '>'


def _total_h(atom: Atom) -> int:
    """原子的总氢数 = 隐氢 + 显式 H 邻居数（ActiveH 计入）。"""
    explicit = sum(1 for bond in atom.bonds if bond.other(atom).name == 'h')
    return atom.implicit_h + explicit


def _atom_pi_system(atom: Atom) -> PiSystem | None:
    """原子所属的 π 体系（validate 保证至多一个）；未参与时返回 None。"""
    for pi in atom.belong.pi_systems:
        if atom in pi.atoms:
            return pi
    return None


def _atom_matches_pattern(atom: Atom, p_atom: PatternAtom) -> bool:
    """目标原子是否满足模式原子的元素 / 总氢 / π 约束。"""
    if p_atom.element is not None and atom.name != p_atom.element:
        return False
    total_h = _total_h(atom)
    if p_atom.h is not None and total_h != p_atom.h:
        return False
    if p_atom.h_min is not None and total_h < p_atom.h_min:
        return False
    if p_atom.pi is True and _atom_pi_system(atom) is None:
        return False
    if p_atom.pi is False and _atom_pi_system(atom) is not None:
        return False
    return True


def _pattern_bond_between(
    pattern: Pattern,
    p1: PatternAtom,
    p2: PatternAtom,
) -> PatternBond | None:
    """两个模式原子之间的模式键（模式保证至多一条）。"""
    pair = frozenset((p1, p2))
    for bond in pattern.bonds:
        if frozenset((bond.atom1, bond.atom2)) == pair:
            return bond
    return None


def _order_matches(
    pattern_order: int | tuple[int, ...] | None,
    target_order: int,
) -> bool:
    if pattern_order is None:
        return True
    if isinstance(pattern_order, int):
        return pattern_order == target_order
    return target_order in pattern_order


def _check_pattern(pattern: Pattern) -> None:
    """校验模式合法性；非法抛 ValueError。"""
    if not pattern.atoms:
        raise ValueError("模式至少需要一个原子")
    if len(set(pattern.atoms)) != len(pattern.atoms):
        raise ValueError("模式中存在重复原子")
    atom_set = set(pattern.atoms)
    seen_pairs: set[frozenset[PatternAtom]] = set()
    for bond in pattern.bonds:
        if bond.atom1 not in atom_set or bond.atom2 not in atom_set:
            raise ValueError("模式键的端点必须属于该模式")
        if bond.atom1 is bond.atom2:
            raise ValueError("模式键不能连接同一个原子")
        pair = frozenset((bond.atom1, bond.atom2))
        if pair in seen_pairs:
            raise ValueError("同一对模式原子之间只能有一条模式键")
        seen_pairs.add(pair)
        _check_pattern_order(bond.order)


def _find_pattern_benzene_rings(pattern: Pattern) -> set[frozenset[PatternAtom]]:
    """检测模式中的凯库勒式苯环，返回环内键的端点对集合。

    仅精确键级（int）参与交替判定；命中的环内键在匹配时统一按单键处理，
    使凯库勒式苯环模式与隐式（单键 + π 体系）目标互相匹配。
    非苯环双键（如羰基 C=O）不受影响，仍严格匹配。
    """
    adjacency: dict[PatternAtom, list[tuple[PatternAtom, int]]] = {
        atom: [] for atom in pattern.atoms
    }
    for bond in pattern.bonds:
        if not isinstance(bond.order, int):
            continue
        adjacency[bond.atom1].append((bond.atom2, bond.order))
        adjacency[bond.atom2].append((bond.atom1, bond.order))
    edges: set[frozenset[PatternAtom]] = set()
    seen: set[frozenset[PatternAtom]] = set()
    for start in pattern.atoms:
        if start.element != 'c':
            continue
        path: list[PatternAtom] = [start]
        start_id = id(start)

        def visit() -> None:
            if len(path) == 6:
                last = path[-1]
                closing = next(
                    (order for neighbor, order in adjacency[last] if neighbor is start),
                    None,
                )
                if closing is None:
                    return
                orders: list[int] = []
                for i in range(5):
                    order = next(
                        order for neighbor, order in adjacency[path[i]]
                        if neighbor is path[i + 1]
                    )
                    orders.append(order)
                orders.append(closing)
                # 交替判定：偶数位键级相同、奇数位键级相同，且两者不同
                if orders[0] == orders[1] or any(
                        orders[i] != orders[i % 2] for i in range(6)):
                    return
                ring_set = frozenset(path)
                if ring_set in seen:
                    return
                # 无弦校验：环内每个原子恰好 2 条环内键
                for atom in path:
                    ring_neighbors = sum(
                        1 for neighbor, _ in adjacency[atom]
                        if neighbor in ring_set
                    )
                    if ring_neighbors != 2:
                        return
                seen.add(ring_set)
                for i in range(6):
                    edges.add(frozenset((path[i], path[(i + 1) % 6])))
                return
            current = path[-1]
            for neighbor, _ in adjacency[current]:
                if neighbor is start and len(path) > 1:
                    continue
                if neighbor.element != 'c' or id(neighbor) <= start_id:
                    continue
                if neighbor in path:
                    continue
                path.append(neighbor)
                visit()
                path.pop()

        visit()
    return edges


def find_substructure_matches(
    pattern: Pattern,
    molecule: Molecule,
    *,
    limit: int | None = None,
) -> list[SubstructureMatch]:
    """在分子中查找所有子结构匹配（子结构语义：目标分子允许模式之外的额外键）。

    匹配前先校验模式与分子结构（无效抛 ValueError）；limit 限制返回数量
    （None 表示全部，limit=1 等价于“是否存在”）。结果按命中的目标原子集合去重，
    对称模式（如硝基的两个 O）每个基团位置只算一次。
    """
    if limit is not None and limit < 1:
        raise ValueError("limit 必须 >= 1")
    _check_pattern(pattern)
    pattern_ring_edges = _find_pattern_benzene_rings(pattern)
    molecule.validate()
    if not molecule.atoms:
        return []

    candidates: dict[PatternAtom, list[Atom]] = {
        p_atom: [atom for atom in molecule.atoms if _atom_matches_pattern(atom, p_atom)]
        for p_atom in pattern.atoms
    }
    if any(not cand for cand in candidates.values()):
        return []

    results: list[SubstructureMatch] = []
    seen_images: set[frozenset[Atom]] = set()
    assignment: dict[PatternAtom, Atom] = {}
    used: set[Atom] = set()
    pi_map: dict[int, PiSystem] = {}

    def target_bond_order(atom1: Atom, atom2: Atom) -> int | None:
        for bond in atom1.bonds:
            if bond.other(atom1) is atom2:
                return bond.order
        return None

    def feasible(p_atom: PatternAtom, atom: Atom) -> bool:
        """候选原子是否可与模式原子配对（基于已匹配映射）。"""
        if atom in used:
            return False
        if not _atom_matches_pattern(atom, p_atom):
            return False
        if p_atom.pi_group is not None:
            pi = _atom_pi_system(atom)
            if pi is None:
                return False
            mapped = pi_map.get(p_atom.pi_group)
            if mapped is not None and mapped is not pi:
                return False
        for other_p, other_atom in assignment.items():
            p_bond = _pattern_bond_between(pattern, p_atom, other_p)
            if p_bond is None:
                continue
            p_order = (1 if frozenset((p_atom, other_p)) in pattern_ring_edges
                       else p_bond.order)
            order = target_bond_order(atom, other_atom)
            if order is None or not _order_matches(p_order, order):
                return False
        return True

    def backtrack() -> bool:
        """返回 True 表示已到 limit，提前终止。"""
        if len(assignment) == len(pattern.atoms):
            image = frozenset(assignment.values())
            if image not in seen_images:
                seen_images.add(image)
                results.append(SubstructureMatch(assignment, pi_map, pattern))
                if limit is not None and len(results) >= limit:
                    return True
            return False
        best_p: PatternAtom | None = None
        best_candidates: list[Atom] = []
        for p_atom in pattern.atoms:
            if p_atom in assignment:
                continue
            cand = [atom for atom in candidates[p_atom] if feasible(p_atom, atom)]
            if not cand:
                return False
            if best_p is None or len(cand) < len(best_candidates):
                best_p = p_atom
                best_candidates = cand
                if len(cand) == 1:
                    break
        assert best_p is not None
        for atom in best_candidates:
            added_group: int | None = None
            if best_p.pi_group is not None and best_p.pi_group not in pi_map:
                pi = _atom_pi_system(atom)
                assert pi is not None  # feasible 已保证
                pi_map[best_p.pi_group] = pi
                added_group = best_p.pi_group
            assignment[best_p] = atom
            used.add(atom)
            if backtrack():
                return True
            used.remove(atom)
            del assignment[best_p]
            if added_group is not None:
                del pi_map[added_group]
        return False

    backtrack()
    return results


def molecule_has_substructure(pattern: Pattern, molecule: Molecule) -> bool:
    """分子是否包含至少一个模式匹配。"""
    return bool(find_substructure_matches(pattern, molecule, limit=1))


def copy_molecule(
    molecule: Molecule,
    atom_map: dict[Atom, Atom] | None = None,
) -> Molecule:
    """深拷贝分子：新建独立的原子/键/π 体系，结构与原分子相同。

    拷贝前先校验结构（validate），无效结构抛 ValueError。保留 name、ActiveH 子类、
    键级、π 体系 dbe 与成员顺序；副本 _snapshot 为空（读取时惰性重建），编辑副本
    不影响原分子。atom_map 非 None 时填入“原原子 -> 副本原子”映射（供反应规则等
    按原分子定位副本原子）。
    """
    molecule.validate()
    clone = Molecule(name=molecule.name)
    mapping: dict[Atom, Atom] = {}
    for atom in molecule.atoms:
        copy_atom: Atom
        if isinstance(atom, ActiveH):
            copy_atom = ActiveH(clone)
        else:
            copy_atom = Atom(atom.name, clone)
        copy_atom._charge = atom._charge
        mapping[atom] = copy_atom
    for bond in molecule.bonds:
        atom1, atom2 = bond.atoms
        _make_bond(
            mapping[atom1],
            mapping[atom2],
            order=bond.order,
            stereo=bond.stereo,
            aromatic=bond.aromatic,
        )
    for pi in molecule.pi_systems:
        members = [mapping[atom] for atom in pi.atoms]
        clone.pi_systems.append(_make_pi_system(members, dbe=pi.dbe, aromatic=pi.aromatic))
    if atom_map is not None:
        atom_map.clear()
        atom_map.update(mapping)
    return clone


def merge_molecules(
    molecules: Sequence[Molecule],
    atom_map: dict[Atom, Atom] | None = None,
    pi_map: dict[PiSystem, PiSystem] | None = None,
) -> Molecule:
    """将多个分子合并进同一个新分子容器（多反应物反应的工作分子）。

    合并前逐个校验源分子（validate），无效结构抛 ValueError；合并结果与各
    源分子拼接一致，保留 ActiveH 子类、键级、π 体系 dbe 与成员顺序，合并后
    校验一次整体结构。atom_map / pi_map 非 None 时分别填入“源原子 -> 合并
    原子”“源 π 体系 -> 合并 π 体系”映射（供反应引擎定位副本原子）。
    """
    merged = Molecule()
    atom_mapping: dict[Atom, Atom] = {}
    pi_mapping: dict[PiSystem, PiSystem] = {}
    for molecule in molecules:
        molecule.validate()
        for atom in molecule.atoms:
            copy_atom: Atom
            if isinstance(atom, ActiveH):
                copy_atom = ActiveH(merged)
            else:
                copy_atom = Atom(atom.name, merged)
            copy_atom._charge = atom._charge
            atom_mapping[atom] = copy_atom
        for bond in molecule.bonds:
            atom1, atom2 = bond.atoms
            _make_bond(
                atom_mapping[atom1],
                atom_mapping[atom2],
                order=bond.order,
                stereo=bond.stereo,
                aromatic=bond.aromatic,
            )
        for pi in molecule.pi_systems:
            members = [atom_mapping[atom] for atom in pi.atoms]
            new_pi = _make_pi_system(members, dbe=pi.dbe, aromatic=pi.aromatic)
            merged.pi_systems.append(new_pi)
            pi_mapping[pi] = new_pi
    if atom_map is not None:
        atom_map.clear()
        atom_map.update(atom_mapping)
    if pi_map is not None:
        pi_map.clear()
        pi_map.update(pi_mapping)
    merged.validate()
    return merged


def _display_formula(formula: dict[str, int]) -> str:
    parts: list[str] = []
    for element, num in formula.items():
        if num == 0:
            continue
        display = element.capitalize()
        parts.append(display + (str(num) if num != 1 else ''))
    return ''.join(parts)
