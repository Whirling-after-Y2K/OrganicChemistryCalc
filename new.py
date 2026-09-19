# 分子构建脚本（由 oc_io 生成）
# 警告：加载本文件会执行其中的代码，请勿运行来源不明的文件。
import organic_chemistry as oc

molecule = oc.Molecule(name='new')

a0 = oc.Atom('c', molecule)



molecule.validate()

if __name__ == "__main__":
    print(molecule)
