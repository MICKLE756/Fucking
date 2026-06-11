# REPRODUCE.md — 复现指南 (Reproducibility)

本文件给出**从零复现**主结果与消融实验所需的：环境、数据布局、确定性设置、逐条命令、期望输出格式。
配套的日常运行说明见 [`RUN.md`](RUN.md)；本文件聚焦“**可复现 + 可消融**”。

---

## 1. 环境 (Environment)

```bash
# 方式 A：conda
conda env create -f environment.yml
conda activate mecpe

# 方式 B：已有 Python(3.10+) 环境
pip install -r requirements.txt
```

- `torch` 请按你的 CUDA 选择对应 wheel（见 `environment.yml` 注释；CPU-only 也能跑单元测试）。
- 不再依赖 `attrdict`（该包在 Python ≥3.10 已损坏）——已用 `src/tools.py::AttrDict` 内置替代。

**冒烟测试（无需 GPU / 无需数据集，只验证代码逻辑）：**
```bash
python -m tests.test_method
python -m tests.test_model_integration
```
两条都应以 `... tests passed.` 结束。

---

## 2. 数据布局 (Data layout)

训练/评估走标准 ECF（MECPE）划分。`src/config.yaml` 里的路径字段指向数据目录；默认形如：

```
<data_root>/
├── train.txt        # 1001 篇对话（gold 情绪 + 情绪-原因对）
├── dev.txt          # 验证集
├── test.txt         # 测试集
├── audio_embedding_*.npy / video_embedding_*.npy   # 多模态特征
└── ...
```

> 数据集本身不随仓库分发。把上述文件放好后，确认 `src/config.yaml` 中的 `*_dir` / `*_file` 指向正确路径。
> 离线 LLM 老师标注 (`llm_anno_path`) 是**可选**的；为 `null` 时蒸馏项自动跳过（详见 `RUN.md`）。

---

## 3. 确定性 (Determinism)

- 单一种子入口：`--seed`（或 `src/config.yaml: seed`）。`src/tools.py::set_seed` 统一固定
  `torch / cuda / numpy / random / PYTHONHASHSEED`，并设 `cudnn.deterministic`、
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- 论文报数请用**多种子均值±方差**（见 §6），不要只报单次最好值。

---

## 4. 评估协议 (Evaluation protocol) —— 重要

`src/config.yaml: eval_threshold_on` 控制判定阈值的选取：

| 取值 | 含义 | 用途 |
|---|---|---|
| `valid`（默认） | 阈值在**验证集**上选，再**固定**应用到测试集 | **论文/对外报数**：无测试集内调参泄漏 |
| `test` | 在测试集上直接搜阈值（旧行为） | 仅复现历史数字（会偏高、不可对外声称） |

> 之前的 `Graph(55.29)` 数字是在 `test` 协议下得到的。对外投稿请用默认 `valid` 协议重跑主表，
> 这会让数字更诚实（通常略低），审稿人不会因“在测试集上调阈值”扣分。

---

## 5. 训练主模型 (Train the full model)

```bash
# 完整方法（所有组件开启 = 默认）
python main.py --seed 0

# 旧版 Graph 基线（关闭整套 method 通路）
python main.py --seed 0 --set use_method=no
```

训练结束会自动加载最佳 checkpoint 并在测试集上打印一段结果（即下面“期望输出格式”）。

---

## 6. 消融 + 多种子 (Ablations + multi-seed)

所有组件都有独立开关（默认全开；`--set X=no` 关掉单个组件做消融）：

| 开关 | 方法章节 | 关闭后 |
|---|---|---|
| `use_self_loop_fix` | §3.3 | 去掉对角自环签名 |
| `use_emotion_transition` | §3.3 | 去掉情绪转移关系边 |
| `use_necessity` | §3.5 | 去掉扰动必要性证据 z^nec |
| `use_pos_prior` | §3.6 | 去掉有界位置先验 b^pos |
| `use_distillation` | §3.7 | 融合中去掉 LLM 证据 z^rea 且不算 L_dst |
| `fusion_mode` | §3.8 | `gated`=退化为无条件门（对照 `cond`） |

单条示例：
```bash
python main.py --seed 0 --set use_necessity=no
python main.py --seed 1 --set fusion_mode=gated --set epoch_size=25
```

一键跑“完整 + 每个消融 × N 种子”，再汇总成 mean±std 表：
```bash
bash scripts/run_ablations.sh "0 1 2" 25      # 种子 0/1/2，每次 25 epoch
python scripts/aggregate_results.py results/logs -o results/ablation_table
# -> 终端打印 markdown 表，并写出 results/ablation_table.md / .csv
```

---

## 7. 期望输出格式 (Expected output format)

测试集最终结果形如（数字仅为格式示例，非真实结果）：

```
Pair Pre. 53.1234   Rec. 57.8910   F1 55.4000   (th=0.120)
TP 1234   Pred. 2300   Gold. 2100
Emo:   Pre. ...  Rec. ...  F1 70.1234
Cause: Pre. ...  Rec. ...  F1 68.5678
```

`scripts/aggregate_results.py` 解析每个日志的**最后一段** `Pair/Emo/Cause` 指标
（即 `final_evaluate` 打印的测试结果），按 config 聚合多种子的均值±方差。

---

## 8. 主表 / 消融表模板 (Tables)

可直接填的模板见 [`results/TABLES.md`](results/TABLES.md)：包含
（a）主表：本方法 vs baseline；（b）消融表：行标签与上面的开关一一对应。
