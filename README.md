# Einsummable

本仓库存放 Einsummable 项目的全部材料。

## 目录

- `docs/programming_project.pdf` — 项目说明（题目要求）
- `harness/` — 成对平方距离（Mahalanobis）内核的基准测试框架：朴素基线、BLAS 参考实现、正确性检查、计时、网格运行脚本与绘图脚本。构建与使用方法见 [`harness/README.md`](harness/README.md)。

## 快速开始

```sh
cd harness
scons
./bench --list
```
