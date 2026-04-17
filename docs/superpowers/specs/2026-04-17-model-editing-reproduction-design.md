# Model Editing Research Platform Specification

## Project Overview

**Goal:** 构建一个模型编辑论文复现与 idea 尝试的研究平台，支持 7B-9B 大模型的多种编辑方法（定位+编辑、梯度优化、提示编辑、终身编辑）。

**模型选择：**
- Qwen2.5-7B-Instruct（主）
- Qwen2.5-9B-Instruct（备）

**复现论文清单：**

| # | 论文 | 方法类别 |
|---|-----|---------|
| 1 | ROME (2022) | 定位+编辑 |
| 2 | MEMIT (2023) | 定位+编辑 |
| 3 | MEND (2022) | 梯度优化 |
| 4 | GRAD (2023) | 梯度优化 |
| 5 | KE-Tuning (2023) | 提示编辑 |
| 6 | TIES-Merging (2023) | 提示编辑 |
| 7 | LiveEdit (2024) | 终身编辑 |

---

## Project Structure

```
/home/disk1/jxy/Model_Editing/
├── README.md
├── requirements.txt
├── configs/                    # 配置文件
│   ├── model/                  # 模型配置
│   │   ├── qwen2.5-7b.yaml
│   │   └── qwen2.5-9b.yaml
│   └── experiment/             # 实验配置
│       └── reproduce.yaml
├── datasets/                   # 数据集
│   ├── zsre/                   # 论文常用测试集
│   ├── counterfact/            # CounterFact 基准
│   └── livedit/                # LiveEdit 终身编辑数据
├── src/                        # 核心代码
│   ├── models/                 # 模型封装
│   │   └── qwen_model.py
│   ├── editors/               # 编辑器实现
│   │   ├── base.py            # 编辑器基类
│   │   ├── rome.py             # ROME
│   │   ├── memit.py            # MEMIT
│   │   ├── mend.py             # MEND
│   │   ├── grad.py             # GRAD
│   │   ├── ke_tuning.py        # KE-Tuning
│   │   ├── ties_merging.py     # TIES-Merging
│   │   └── livedit.py          # LiveEdit
│   ├── evaluation/            # 评估模块
│   │   ├── evaluate.py        # 通用评估
│   │   └── metrics.py         # 评估指标
│   └── utils/                 # 工具函数
│       ├── model_utils.py
│       └── data_utils.py
├── scripts/                   # 脚本
│   ├── train/                 # 训练脚本
│   ├── evaluate/              # 评估脚本
│   └── data/                  # 数据处理脚本
├── notebooks/                 # Jupyter notebooks
│   ├── exploration/           # 探索性实验
│   └── analysis/              # 结果分析
├── logs/                      # 日志
├── checkpoints/               # 模型checkpoint
└── docs/                      # 文档
    ├── superpowers/
    └── papers/                # 论文笔记
```

---

## Core Components

### 1. Editor Base Class

所有编辑方法继承自 `base.py` 中的 `BaseEditor` 类：

```python
class BaseEditor(ABC):
    def edit(self, model, request: EditRequest) -> EditResult
    def batch_edit(self, model, requests: List[EditRequest]) -> List[EditResult]
    def evaluate(self, model, dataset) -> EvaluationResult
```

### 2. Model Wrapper

`qwen_model.py` 封装 Qwen 模型的加载与推理：

```python
class QwenModel:
    def load(model_name: str, device: str)
    def generate(prompt: str) -> str
    def get_hidden_states(layer_idx: int, token_idx: int) -> Tensor
    def set_hidden_states(layer_idx: int, token_idx: int, value: Tensor)
```

### 3. Evaluation Framework

统一的评估接口，支持：
- Efficacy（编辑效果）
- Genericity（泛化能力）
- Locality（局部性）
- Portability（可移植性）

---

## Implementation Priority

**Phase 1: 基础架构**
- 项目结构搭建
- Model Wrapper 实现
- Base Editor 类定义
- 评估框架搭建

**Phase 2: 定位+编辑方法**
- ROME 实现
- MEMIT 实现

**Phase 3: 梯度优化方法**
- MEND 实现
- GRAD 实现

**Phase 4: 提示编辑方法**
- KE-Tuning 实现
- TIES-Merging 实现

**Phase 5: 终身编辑**
- LiveEdit 实现

---

## Hardware Requirements

- 双卡 A6000 48G
- 预计显存：7B 模型单卡可加载，9B 模型建议多卡或量化
- 建议使用 DeepSpeed ZeRO-3 for 9B 模型
