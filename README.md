# LM Lab：亲手实现语言模型

这个仓库用于实现、理解和验证语言模型的关键组件。路线为 **Basics → Systems → Scaling → Data → Post-training**，推理、评测、数值稳定性与实验复现贯穿全程。完整目标、依赖和验收见 [学习路线](docs/roadmap.md)。

当前只详细搭建 tokenizer 阶段：核心接口有文档和 TODO，算法函数明确抛出 `NotImplementedError`，由你亲手完成。助手负责解释、局部提示、验收与调试。当前规范是本仓库的教学约定；严格匹配官方作业需要对应版本的 handout。

## 安装：Windows PowerShell

需要 Python 3.11+。运行依赖只有标准库，开发依赖为 pytest；初始化和 tokenizer 练习均可在 CPU 上完成。

```powershell
cd C:\Users\liangqianxing\Desktop\llm
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

直接调用虚拟环境的 Python，无须激活环境或修改 PowerShell 执行策略。后续重新打开终端也使用 `.\.venv\Scripts\python.exe`。上面的安装会获取少量开发/构建依赖，不会下载语料或模型。

本次初始化受网络限制，实际创建的 `.venv` 复用了本机已有的 pytest 与 setuptools，并离线安装项目。仅在本机已有兼容依赖时可使用以下替代命令；`--system-site-packages` 会让环境看到全局包，不是完全隔离的安装：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install --no-index --no-deps --no-build-isolation -e ".[dev]"
```

## 📚 文档导航

- **[快速开始](docs/quickstart.md)** - 安装、示例、第一个模型
- **[架构设计](docs/architecture.md)** - 项目结构、技术亮点
- **[API 参考](docs/api_reference.md)** - 完整的 API 文档
- **[学习路线](docs/roadmap.md)** - 里程碑和验收标准
- **[项目总结](PROJECT_SUMMARY.md)** - 完成情况总览

## 今天从哪里开始

### 如果您是初学者

1. **阅读快速开始指南**: [docs/quickstart.md](docs/quickstart.md)
2. **运行分词器测试**: `pytest tests/tokenization/ -v`
3. **尝试示例代码**: 查看 `docs/quickstart.md` 中的示例

### 如果您已完成分词器

1. **学习 Transformer**: 阅读 [docs/architecture.md](docs/architecture.md)
2. **查看 API 文档**: [docs/api_reference.md](docs/api_reference.md)
3. **实现训练流程**: 参考 `docs/quickstart.md` 的训练示例

```powershell
# 初始化检查：结构、安装、导入、语法和 UTF-8
.\.venv\Scripts\python.exe scripts/check_init.py

# T1 功能验收；以后可替换成 t2、t3、t4、t5
.\.venv\Scripts\python.exe -m pytest -m t1 -q

# 全部 tokenizer 功能验收
.\.venv\Scripts\python.exe -m pytest -q
```

**初始化检查通过不等于分词器功能通过。** 初始代码的 T1–T5 尚未实现，功能测试会以 `NotImplementedError` 失败并返回非零退出码；没有 skip 或 xfail。只收集测试也不算功能通过。实现一个任务后，先运行该任务，再运行完整套件检查回归。

本次实际执行的安装、检查和待实现情况见 [初始化检查记录](docs/initialization_check.md)。

| 任务 | 亲手实现的接口 | 验收命令的选择条件 |
| --- | --- | --- |
| T1 | `ByteTokenizer.encode/decode` | `-m t1` |
| T2 | `count_pairs`、`merge_pair`、`train_bpe` 的最小模式 | `-m t2` |
| T3 | `BPETokenizer.encode/decode` | `-m t3` |
| T4 | `pretokenize`、预分词接入、特殊 token 边界 | `-m t4` |
| T5 | `BPETokenizer.save/load`、完整确定性 | `-m t5` |

T3 与部分 T4/T5 测试直接构造手工词表，方便独立调试；集成测试还需要前置任务完成。

## 测量与学习记录

完成相应算法后，用固定的手写小样本做测量：

```powershell
.\.venv\Scripts\python.exe scripts/measure_tokenizer.py --task byte --output runs/tokenizer/byte-001.json
.\.venv\Scripts\python.exe scripts/measure_tokenizer.py --task bpe --vocab-size 280 --pretokenization simple --output runs/tokenizer/bpe-001.json
```

脚本先检查还原和重复编码的一致性，再记录多次计时、token 数、样本指纹、代码版本与硬件。未实现时返回退出码 2，不生成成绩文件；已有输出文件不会被覆盖。样本是教学示例，不代表真实语料，短计时也不能证明普遍性能提升。

复制 [实验模板](docs/experiment_template.md) 到 `runs/<实验名>/record.md`，保存命令和原始输出；可将不含敏感数据的结论整理进 `docs/`。用 `docs/roadmap.md` 的“未开始 / 进行中 / 已验证”记录进展，只有实际实现并通过验收后才标记“已验证”。首次 Git 提交前，代码版本记录为“尚无提交”并注明工作区状态。

## 目录与提交边界

```text
src/lm_lab/tokenization/   算法接口与练习
tests/tokenization/        小规模行为验收
docs/                     学习路线、任务、实验模板与检查记录
scripts/                  初始化检查与测量工具
data/                     本地语料，仅 README 提交
runs/                     本地输出，仅 README 提交
```

数据集、模型权重、训练输出、缓存和密钥均不提交。不要下载现成课程答案，也不要用 tokenizer 库替代自己的 BPE；之后引入成熟库时须标注其对照用途。贡献规范见 [AGENTS.md](AGENTS.md)。
