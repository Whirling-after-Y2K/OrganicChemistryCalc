"""分子储存与加载（构建代码存储，无第三方依赖）。

存储文件是可独立运行的 Python 构建脚本，统一使用 .mol 后缀
（内容本质仍是 Python 代码，换后缀只是避免浏览器把 .py 当作
 可执行脚本下载时弹出警告）：
- 以 `import organic_chemistry as oc` 开头，按"原子 → 键 → π 体系"顺序
  使用公开 API 构建分子，末尾定义顶层变量 `molecule` 并调用 validate()；
- 直接运行脚本会打印分子；加载时执行脚本取回 `molecule` 变量并再次校验。

安全警告：加载即执行代码，请勿加载来源不明的文件。

按项目约定，stereo / aromatic / 形式电荷等暂不处理的属性不写入存储；
dbe 参与不饱和度计算，始终显式写入。

旧版 .py 分子文件不受影响：加载按内容识别，与后缀无关。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import organic_chemistry as oc

_HEADER: str = (
    "# 分子构建脚本（由 oc_io 生成）\n"
    "# 警告：加载本文件会执行其中的代码，请勿运行来源不明的文件。\n"
    "import organic_chemistry as oc\n"
)

# 分子文件后缀：内容仍是 Python 构建脚本，用 .mol 只是避免浏览器对
# .py 文件弹出下载警告。
MOLECULE_SUFFIX: str = ".mol"


def with_molecule_suffix(filename: str) -> str:
    """确保文件名以 .mol 结尾：未写后缀或写了其他后缀（含旧版 .py）时，直接在末尾补 .mol。"""
    name = filename.strip()
    if not name or name.lower().endswith(MOLECULE_SUFFIX):
        return name
    return name + MOLECULE_SUFFIX


def molecule_to_code(molecule: oc.Molecule) -> str:
    """将分子生成为一段可执行的构建代码；保存前强制校验结构。"""
    molecule.validate()
    lines: list[str] = [_HEADER]

    if molecule.name is None:
        lines.append("molecule = oc.Molecule()")
    else:
        lines.append(f"molecule = oc.Molecule(name={molecule.name!r})")
    lines.append("")

    # 原子：按 molecule.atoms 顺序命名为 a0、a1...
    atom_names: dict[oc.Atom, str] = {}
    for index, atom in enumerate(molecule.atoms):
        name = f"a{index}"
        atom_names[atom] = name
        if isinstance(atom, oc.ActiveH):
            lines.append(f"{name} = oc.ActiveH(molecule)")
        else:
            lines.append(f"{name} = oc.Atom({atom.name!r}, molecule)")
    lines.append("")

    # 键：按 molecule.bonds 顺序
    for bond in molecule.bonds:
        atom1, atom2 = bond.atoms
        args = f"{atom_names[atom1]}, {atom_names[atom2]}"
        if bond.order != 1:
            args += f", order={bond.order}"
        lines.append(f"oc.add_bond({args})")
    lines.append("")

    # π 体系：按 molecule.pi_systems 顺序，dbe 显式写出
    for pi in molecule.pi_systems:
        members = ", ".join(atom_names[atom] for atom in pi.atoms)
        lines.append(f"oc.add_pi_system([{members}], dbe={pi.dbe})")
    lines.append("")

    lines.append("molecule.validate()")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    print(molecule)")
    return "\n".join(lines)


def molecule_from_code(code: str) -> oc.Molecule:
    """执行构建代码并重建分子；格式或结构非法抛 ValueError。"""
    namespace: dict[str, Any] = {"__name__": "oc_molecule_build"}
    try:
        exec(code, namespace)
    except Exception as exc:
        raise ValueError(f"构建代码执行失败：{exc}") from exc
    molecule = namespace.get("molecule")
    if not isinstance(molecule, oc.Molecule):
        raise ValueError("构建代码必须定义 molecule 变量（oc.Molecule 实例）")
    molecule.validate()
    return molecule


def save_molecule(molecule: oc.Molecule, path: str | os.PathLike[str]) -> None:
    """将分子保存为 UTF-8 构建脚本文件（惯例后缀 .mol，内容仍是 Python）。

    保存前校验结构，非法结构抛 ValueError；只接受单一连通分量的分子，
    存在互不成键的片段时同样抛 ValueError。
    """
    oc.ensure_single_component(molecule)
    code = molecule_to_code(molecule)
    with open(path, "w", encoding="utf-8") as file:
        file.write(code)
        file.write("\n")


def load_molecule(path: str | os.PathLike[str]) -> oc.Molecule:
    """从 UTF-8 构建脚本文件加载分子（.mol 与旧版 .py 均按内容识别）。

    格式或结构非法抛带文件名的 ValueError。
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            code = file.read()
    except UnicodeDecodeError as exc:
        raise ValueError(f"文件 {path} 不是有效的 UTF-8 文本：{exc}") from exc
    try:
        return molecule_from_code(code)
    except ValueError as exc:
        raise ValueError(f"文件 {path}：{exc}") from exc
