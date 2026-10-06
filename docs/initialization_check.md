# 初始化检查记录

记录日期：2026-10-07。本文记录工程初始化的实际检查，不是 tokenizer 实现或性能实验的成绩。学习里程碑 T1–T5 仍为“未开始”。

## 环境与安装

- 初始工作目录为空，没有根目录 `AGENTS.md`，也没有 Git 仓库；已创建新的贡献指南并执行 `git init`，没有生成提交。
- 已检查上级 `AGENTS.md`；其论文写作流程不属于本次学习项目初始化范围，已有上级文件未修改。
- Python 3.13.5；`lm-lab` 0.1.0；pytest 9.0.1；setuptools 70.2.0。
- 普通联网安装因当前网络限制失败。实际环境采用 `python -m venv --system-site-packages .venv`，复用本机已有依赖，再运行 `pip install --no-index --no-deps --no-build-isolation -e ".[dev]"`，离线构建并安装成功。
- 环境会看到全局包，因此不是完全隔离的依赖环境；常规安装和离线替代命令均在 README 中说明。
- 本次检查使用本机 CPU，未使用 GPU，未下载语料/模型，未启动付费计算。没有做 tokenizer 性能测量。
- 代码版本：尚无 Git 提交；源文件为未提交的新文件。不要为本次初始化编造提交哈希。
- 最终核对时另外发现 `START_HERE.md` 和 `demo_bytes.py`；两者均保留，未修改。本记录的文件完整性检查覆盖本次创建/修改的工程文件。

## 初始化检查与功能验收分开报告

| 检查 | 实际结果 | 含义 |
| --- | --- | --- |
| `scripts/check_init.py` | 退出码 0，所有检查项通过 | 必要文件、配置、虚拟环境、导入、版本、Python 语法、UTF-8 就绪；不执行核心算法 |
| `pytest --collect-only -q` | 退出码 0，收集 80 项 | 测试导入及 marker 配置有效；不代表行为正确 |
| `pytest -q --tb=short --junitxml=runs/setup/tokenization.xml` | 退出码 1，80 项失败 | 全部失败原因都是核心函数的 `NotImplementedError`，功能验收尚未通过 |
| `measure_tokenizer.py --help` | 退出码 0 | CLI 参数说明可用 |
| README 的 byte / bpe 测量命令 | 两者均退出码 2，报告 PENDING | 未实现，未写入任何测量成绩文件 |
| Git 忽略规则核对 | 通过 | 数据、运行输出、环境、密钥及权重路径被忽略；代码和两个目录 README 可跟踪 |
| `chinese-text-integrity` 检查 | 退出码 0 | 中文文档、代码和测试未发现编码损坏 |

功能失败明细：T1 为 16 项、T2 为 21 项、T3 为 13 项、T4 为 20 项、T5 为 10 项。没有测试收集错误、skip 或 xfail，也没有其他失败原因。测试使用正常断言与指定异常，不将待实现转换为成功。

## 原始输出

- `runs/setup/init.txt`：初始化检查输出。
- `runs/setup/tokenization.txt`：完整功能测试输出。
- `runs/setup/tokenization.xml`：逐项失败类型与栈信息。
- `runs/setup/acceptance_summary.json`：从上述 XML 统计的失败分类。
- `runs/setup/measurement_status.txt`：两个测量入口的待实现报告。

这些是初始化诊断日志，被 Git 忽略；不是训练或性能成绩。正式实现后，应以新运行记录为准，不覆写本文中的历史检查结果。

## 接下来的三个任务

1. 完成 `ByteTokenizer.encode/decode`，运行 `python -m pytest -m t1 -q`。
2. 在 `bpe.py` 完成 `count_pairs` 与 `merge_pair`，分别运行 `python -m pytest -m t2 -k adjacent_pair_counts -q` 和 `python -m pytest -m t2 -k non_overlapping_merge -q`。
3. 完成 `train_bpe` 的最小 ASCII 模式，先手算 `abab` 和 `abac`，再运行 `python -m pytest -m t2 -q`。

以上 `python` 请使用 README 中的 `.\.venv\Scripts\python.exe`。实现后填写实验模板，并根据实际验收结果更新路线状态。
