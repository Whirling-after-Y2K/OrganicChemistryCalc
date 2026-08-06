"""分子储存与加载（JSON 序列化，无第三方依赖）。

格式版本 1：
{
  "format": 1,
  "name": "乙烷",
  "atoms": [{"element": "c", "active_h": false}, ...],
  "bonds": [{"a": 0, "b": 1, "order": 1, "stereo": null, "aromatic": false}, ...],
  "pi_systems": [{"atoms": [0, 1, ...], "dbe": 3, "aromatic": false}, ...]
}

- 原子用 0 基下标互相引用；active_h 仅对元素 h 允许为 true（ActiveH 节点）。
- 保存前强制 molecule.validate()，非法结构拒绝写入；加载采用严格失败策略，
  格式或结构非法一律抛 ValueError。
"""

from __future__ import annotations

import json
import os
from typing import Any

import organic_chemistry as oc

FORMAT_VERSION: int = 1

# 每层结构必需的键；多余键忽略（为未来兼容）
_TOP_KEYS: tuple[str, ...] = ("format", "atoms", "bonds", "pi_systems")
_ATOM_KEYS: tuple[str, ...] = ("element", "active_h")
_BOND_KEYS: tuple[str, ...] = ("a", "b", "order")
_PI_KEYS: tuple[str, ...] = ("atoms",)


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], path: str) -> None:
    """要求 data 包含全部必需键，缺一即抛 ValueError。"""
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{path}: 缺少必需键 {missing!r}")


def molecule_to_dict(molecule: oc.Molecule) -> dict[str, Any]:
    """将分子序列化为可 JSON 化的 dict；保存前强制校验结构。"""
    molecule.validate()
    atom_index: dict[oc.Atom, int] = {
        atom: index for index, atom in enumerate(molecule.atoms)
    }
    atoms: list[dict[str, Any]] = [
        {"element": atom.name, "active_h": isinstance(atom, oc.ActiveH)}
        for atom in molecule.atoms
    ]
    bonds: list[dict[str, Any]] = [
        {
            "a": atom_index[bond.atoms[0]],
            "b": atom_index[bond.atoms[1]],
            "order": bond.order,
            "stereo": bond.stereo,
            "aromatic": bond.aromatic,
        }
        for bond in molecule.bonds
    ]
    pi_systems: list[dict[str, Any]] = [
        {
            "atoms": [atom_index[atom] for atom in pi.atoms],
            "dbe": pi.dbe,
            "aromatic": pi.aromatic,
        }
        for pi in molecule.pi_systems
    ]
    return {
        "format": FORMAT_VERSION,
        "name": molecule.name,
        "atoms": atoms,
        "bonds": bonds,
        "pi_systems": pi_systems,
    }


def _load_atoms(molecule: oc.Molecule, raw_atoms: Any) -> None:
    """按索引顺序重建原子；ActiveH 由 active_h 标志区分。"""
    if not isinstance(raw_atoms, list):
        raise ValueError("atoms 必须是数组")
    for index, raw in enumerate(raw_atoms):
        path = f"atoms[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: 必须是对象")
        _require_keys(raw, _ATOM_KEYS, path)
        element = raw["element"]
        if not isinstance(element, str):
            raise ValueError(f"{path}.element: 必须是字符串")
        element = element.lower()
        if element not in oc.CHEMISTRY_BOND_DICT:
            raise ValueError(f"{path}.element: 未知元素 {element!r}")
        active_h = raw["active_h"]
        if not isinstance(active_h, bool):
            raise ValueError(f"{path}.active_h: 必须是布尔值")
        if active_h and element != "h":
            raise ValueError(f"{path}: active_h 仅允许用于元素 h")
        if active_h:
            oc.ActiveH(molecule)
        else:
            oc.Atom(element, molecule)


def _load_bonds(molecule: oc.Molecule, raw_bonds: Any) -> None:
    """重建键；显式拒绝自环与同一原子对重复成键（平行键）。"""
    if not isinstance(raw_bonds, list):
        raise ValueError("bonds 必须是数组")
    atoms = molecule.atoms
    seen_pairs: set[tuple[int, int]] = set()
    for index, raw in enumerate(raw_bonds):
        path = f"bonds[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: 必须是对象")
        _require_keys(raw, _BOND_KEYS, path)
        a = raw["a"]
        b = raw["b"]
        if type(a) is not int or type(b) is not int:
            raise ValueError(f"{path}: a/b 必须是整数下标")
        if not 0 <= a < len(atoms) or not 0 <= b < len(atoms):
            raise ValueError(f"{path}: 原子下标越界")
        if a == b:
            raise ValueError(f"{path}: 原子不能与自身成键")
        pair = (min(a, b), max(a, b))
        if pair in seen_pairs:
            raise ValueError(f"{path}: 同一原子对重复成键（平行键）")
        seen_pairs.add(pair)
        order = raw["order"]
        if type(order) is not int or not 1 <= order <= oc.MAX_BOND_ORDER:
            raise ValueError(f"{path}.order: 必须是 1-{oc.MAX_BOND_ORDER} 的整数")
        stereo = raw.get("stereo")
        if stereo is not None and not isinstance(stereo, str):
            raise ValueError(f"{path}.stereo: 必须是字符串或 null")
        aromatic = raw.get("aromatic", False)
        if not isinstance(aromatic, bool):
            raise ValueError(f"{path}.aromatic: 必须是布尔值")
        oc._make_bond(atoms[a], atoms[b], order=order, stereo=stereo, aromatic=aromatic)


def _load_pi_systems(molecule: oc.Molecule, raw_pis: Any) -> None:
    """重建 π 体系；成员数与下标在加载时校验。"""
    if not isinstance(raw_pis, list):
        raise ValueError("pi_systems 必须是数组")
    atoms = molecule.atoms
    for index, raw in enumerate(raw_pis):
        path = f"pi_systems[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: 必须是对象")
        _require_keys(raw, _PI_KEYS, path)
        raw_members = raw["atoms"]
        if not isinstance(raw_members, list):
            raise ValueError(f"{path}.atoms: 必须是数组")
        if len(raw_members) < 2:
            raise ValueError(f"{path}: π 体系至少需要两个原子")
        members: list[oc.Atom] = []
        for member_index, value in enumerate(raw_members):
            if type(value) is not int:
                raise ValueError(f"{path}.atoms[{member_index}]: 必须是整数下标")
            if not 0 <= value < len(atoms):
                raise ValueError(f"{path}.atoms[{member_index}]: 原子下标越界")
            members.append(atoms[value])
        dbe = raw.get("dbe")
        if dbe is not None and type(dbe) is not int:
            raise ValueError(f"{path}.dbe: 必须是整数或省略")
        aromatic = raw.get("aromatic", False)
        if not isinstance(aromatic, bool):
            raise ValueError(f"{path}.aromatic: 必须是布尔值")
        molecule.pi_systems.append(
            oc._make_pi_system(members, dbe=dbe, aromatic=aromatic)
        )


def molecule_from_dict(data: dict[str, Any]) -> oc.Molecule:
    """从 JSON dict 重建分子；格式或结构非法抛 ValueError。"""
    if not isinstance(data, dict):
        raise ValueError("分子数据必须是 JSON 对象")
    _require_keys(data, _TOP_KEYS, "分子数据")
    if data["format"] != FORMAT_VERSION:
        raise ValueError(
            f"不支持的格式版本 {data['format']!r}，当前仅支持 {FORMAT_VERSION}"
        )
    name = data.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("name 必须是字符串或 null")

    molecule = oc.Molecule(name=name)
    _load_atoms(molecule, data["atoms"])
    _load_bonds(molecule, data["bonds"])
    _load_pi_systems(molecule, data["pi_systems"])
    molecule.validate()
    return molecule


def save_molecule(molecule: oc.Molecule, path: str | os.PathLike[str]) -> None:
    """将分子保存为 UTF-8 JSON 文件（保存前校验结构，非法结构抛 ValueError）。"""
    data = molecule_to_dict(molecule)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_molecule(path: str | os.PathLike[str]) -> oc.Molecule:
    """从 UTF-8 JSON 文件加载分子；格式或结构非法抛带文件名的 ValueError。"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"文件 {path} 不是有效的 JSON：{exc}") from exc
    try:
        return molecule_from_dict(data)
    except ValueError as exc:
        raise ValueError(f"文件 {path}：{exc}") from exc