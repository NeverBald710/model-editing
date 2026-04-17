# Model Editing Research Platform - 项目结构

```
/home/disk1/jxy/Model_Editing/
├── README.md
├── requirements.txt
├── pyproject.toml                    # 项目配置
│
├── configs/                          # 配置文件
│   ├── model/
│   │   ├── qwen2.5-7b.yaml
│   │   └── qwen2.5-9b.yaml
│   ├── editor/
│   │   ├── rome.yaml
│   │   ├── memit.yaml
│   │   ├── mend.yaml
│   │   └── livedit.yaml
│   └── experiment/
│       └── benchmark.yaml
│
├── datasets/                          # 数据集
│   ├── zsre/                         # ZsRE 问答编辑
│   │   └── preprocess.py
│   ├── counterfact/                  # CounterFact 事实编辑
│   │   └── preprocess.py
│   ├── gptje_data/                   # GPT-JE 终身编辑
│   │   └── preprocess.py
│   └── livedit/                      # LiveEdit 专用
│       └── preprocess.py
│
├── src/                              # 核心代码
│   ├── __init__.py
│   ├── models/                       # 模型封装
│   │   ├── __init__.py
│   │   └── qwen_model.py
│   ├── editors/                     # 编辑器实现
│   │   ├── __init__.py
│   │   ├── base.py                  # 基类
│   │   ├── locate_editors/          # 定位+编辑
│   │   │   ├── __init__.py
│   │   │   ├── rome.py
│   │   │   └── memit.py
│   │   ├── gradient_editors/        # 梯度优化
│   │   │   ├── __init__.py
│   │   │   ├── mend.py
│   │   │   └── grad.py
│   │   ├── prompt_editors/          # 提示编辑
│   │   │   ├── __init__.py
│   │   │   ├── ke_tuning.py
│   │   │   └── ties_merging.py
│   │   └── lifelong_editors/        # 终身编辑
│   │       ├── __init__.py
│   │       └── livedit.py
│   ├── evaluation/                  # 评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       ├── model_utils.py
│       ├── data_utils.py
│       └── training_utils.py
│
├── scripts/                        # 脚本
│   ├── prepare_data.sh              # 数据准备
│   ├── run_editor.py               # 运行单个编辑器
│   ├── run_benchmark.py             # 运行基准测试
│   └── analyze_results.py           # 结果分析
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_explore_model.ipynb      # 模型探索
│   ├── 02_reproduce_rome.ipynb      # ROME复现
│   ├── 03_compare_editors.ipynb     # 编辑器对比
│   └── 04_analysis.ipynb           # 结果分析
│
├── results/                        # 实验结果
│   ├── rome/
│   ├── memit/
│   └── livedit/
│
├── logs/                           # 日志
├── checkpoints/                    # 保存的模型checkpoint
│
├── tests/                          # 测试
│   ├── __init__.py
│   ├── test_models/
│   ├── test_editors/
│   └── test_evaluation/
│
└── docs/                           # 文档
    ├── superpowers/
    ├── papers/                      # 论文笔记
    └── progress.md                 # 研究进度
```

## 目录说明

| 目录 | 内容 |
|------|------|
| `configs/` | 模型、编辑器、实验配置（YAML格式） |
| `datasets/` | 各数据集及其预处理脚本（ZsRE, CounterFact, LiveEdit等） |
| `src/models/` | Qwen 模型封装，含推理和隐状态操作 |
| `src/editors/locate_editors/` | ROME, MEMIT — 定位因果链后编辑 |
| `src/editors/gradient_editors/` | MEND, GRAD — 利用梯度信息编辑 |
| `src/editors/prompt_editors/` | KE-Tuning, TIES-Merging — 提示层面编辑 |
| `src/editors/lifelong_editors/` | LiveEdit — 终身/持续编辑 |
| `src/evaluation/` | 统一评估框架（Efficacy/Portability/Locality） |
| `notebooks/` | Jupyter 分析笔记（模型探索→复现→对比→分析） |
| `results/` | 各编辑器实验结果 |
| `tests/` | 单元测试 |

## 编辑器分类

### 定位+编辑 (Locate-then-Edit)
- **ROME** - Rank-One Model Editing
- **MEMIT** - Memory-Based Model Editing

### 梯度优化 (Gradient-Based)
- **MEND** - Hypernetwork-based gradient decomposition
- **GRAD** - Attention-based gradient editing

### 提示编辑 (Prompt-Based)
- **KE-Tuning** - Knowledge Editor Training
- **TIES-Merging** - Task Vector Merging

### 终身编辑 (Lifelong Editing)
- **LiveEdit** - Retrieval-Augmented Continuous Prompt Learning

## 数据集

| 数据集 | 用途 | 论文 |
|--------|------|------|
| ZsRE | 问答式编辑 | - |
| CounterFact | 事实性编辑 | - |
| LiveEdit | 终身编辑基准 | Liu et al. 2024 |
