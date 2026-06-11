# RUN.md — 运行操作指南

精简后的端到端流程。所有命令默认在 `Graph/` 目录下、`conda activate hilo` 环境中运行；
vLLM 单独用 `conda activate /root/autodl-tmp/conda_dirs/vllm_env`。

> 复现/消融的专用说明见 [`REPRODUCE.md`](REPRODUCE.md)（环境、数据布局、确定性、评估协议、消融脚本）。
> 本文件是**日常运行**版；想直接复现主表/消融，照 `REPRODUCE.md` 走即可。

---

## 0. 目录速览（清理后）

```
Graph/
├── main.py                 # 训练 + 评估入口（一条命令跑全程）
├── requirements.txt
├── src/
│   ├── config.yaml         # 主配置（改一行 llm_anno_path 即可开/关 LLM 蒸馏）
│   ├── config_lite.yaml    # 轻量配置（小规模/调试）
│   ├── layer.py            # 底层模块（LSTM / Biaffine / 注意力 / 融合门 / ERGAT）
│   ├── loader.py           # 数据集 + collate
│   ├── method.py           # Method 章组件（证据/蒸馏/自环表征修复 e_self 等）
│   ├── model.py            # 主模型 TextClassification
│   ├── tools.py            # 配置/优化器/随机种子
│   ├── trainer.py          # 训练循环 + 评估
│   ├── annotate_llm.py     # 【离线·老师】全量 LLM 推理标注 → llm_anno.pkl
│   ├── annotate_llm_test.py# 【离线·体检】抽样标注 + 对照 gold 出质量报告
│   ├── calibrate_anno.py   # 【离线·可选】看偏差 / 朝 gold 软校正
│   └── build_selfcause_sft.py # 【进阶·可选】从 gold 生成自环 SFT 微调数据 (LLaMA-Factory)
├── environment.yml         # conda 环境（可复现）
├── requirements.txt        # 精简后的依赖（按真实 import）
├── REPRODUCE.md            # 复现/消融指南
├── RUN.md                  # 本文件
├── scripts/                # run_ablations.sh + aggregate_results.py（消融/多种子）
├── results/                # 主表/消融表模板 + 审计文档（logs/ 不入库）
└── tests/                  # 模块单测 + 集成测试
```

> LLM 只在**离线标注**时用一次，训练/推理阶段**绝不**调用 LLM。

---

## 1. 装环境 + 自测（无需 GPU/数据，先确认逻辑没坏）

```bash
conda activate hilo                      # 服务器现成环境
# 或全新环境：conda env create -f environment.yml && conda activate mecpe
pip install -r requirements.txt          # 已用 hilo 时这步可跳过
python -m tests.test_method && python -m tests.test_model_integration
```

两个都打印 `... passed.` 即通过。（这步**不需要 GPU/数据**，只验证代码逻辑。）

---

## 2.（可选，强烈建议）抽样体检 LLM 老师质量

先在**另一个终端**起 vLLM，别关：

```bash
conda activate vllm_env
vllm serve /root/autodl-tmp/SFT/Causal_LLM --port 8000
```

再抽 50 篇做质量报告（**务必带 `--max-new-tokens 512`**，否则 CoT 被截断 → 全 0.5 兜底）：

```bash
python -m src.annotate_llm_test \
    --train /root/autodl-tmp/Graph/data/dataset/train.txt \
    --vllm-url http://localhost:8000/v1 \
    --model /root/autodl-tmp/SFT/Causal_LLM \
    --k 5 --num 50 --seed 0 --max-new-tokens 512 \
    --out data/llm_anno_sample50.pkl
```

报告重点看：
- `mean rho`（可靠度）应明显 > 0；若接近 0，多半是 token 太小或服务异常。
- `gold self-cause` / `gold cross-utt` 两行的 `s_dir` 高、`s_spur` 低为好。
- `top-|gold|` 命中率：自环、跨句分别看。

---

## 3. 全量离线标注（只标 train，跑一次）

```bash
python -m src.annotate_llm \
    --train /root/autodl-tmp/Graph/data/dataset/train.txt \
    --vllm-url http://localhost:8000/v1 \
    --model /root/autodl-tmp/SFT/Causal_LLM \
    --k 5 --max-new-tokens 512 --batch-size 64 \
    --out data/llm_anno.pkl
```

要点：
- `--max-new-tokens 512`：全量脚本默认 256，对 CoT 偏紧，**必须显式调大**。
- `--batch-size`：并发请求数（默认 32），GPU 够就调大（如 64/128）提速。
- 断点续跑：脚本默认每 100 对存一次 checkpoint，中断后**重跑同一条命令**会自动续；要从头来加 `--no-resume`。
- **绝不标注 valid/test**（会泄漏），只标 train。

---

## 4.（可选）看偏差 / 朝 gold 软校正

```bash
# 只看偏差报告（不改文件）
python -m src.calibrate_anno --pkl data/llm_anno.pkl \
    --train /root/autodl-tmp/Test/hilo/data/dataset/train.txt

# 朝 gold 软锚定，输出新文件（不覆盖原始）
python -m src.calibrate_anno --pkl data/llm_anno.pkl \
    --train /root/autodl-tmp/Test/hilo/data/dataset/train.txt \
    --alpha 0.3 --inject-missing-gold --out data/llm_anno_calib.pkl
```

`--alpha` 越大越贴 gold（0=不动，建议 0.2~0.4）。用了校正就让步骤 5 指向 `_calib` 文件。

---

## 5. 改 `src/config.yaml` 一行

```yaml
llm_anno_path: data/llm_anno.pkl      # 默认 null=关蒸馏；用了校正就写 data/llm_anno_calib.pkl
```

- `null` → 跳过 LLM 蒸馏，但**自环表征修复 `e_self` 仍生效**（在模型里，自动带上）。
- 顺序不能反：**先生成 pkl 再设 `llm_anno_path`**，否则蒸馏自动跳过。

其它常用字段（一般不用动）：`use_method: yes`、`warmup_K: 3`（前 3 轮热身后才启用证据/蒸馏）、`top_m: 50`（每篇证据预算）、`lambda1`（蒸馏权重）。

---

## 6. 训练 + 评估（一条命令）

```bash
python main.py                          # 完整方法（所有组件默认全开）
python main.py --seed 0                 # 固定种子（多种子复现用 --seed 0/1/2...）
python main.py --set use_method=no      # 旧版 Graph 基线对照
```

只跑这一条：过 `warmup_K` 轮且 `llm_anno.pkl` 存在时，蒸馏自动启用。
训练结束自动加载最佳 checkpoint，并在**测试集**打印 `Pair / Emo / Cause` 的 P/R/F1。

- `--seed N`：覆盖配置里的随机种子（多种子跑均值±方差）。
- `--set 键=值`：行内覆盖任意 `config.yaml` 字段，可叠加，如
  `--set epoch_size=25 --set use_necessity=no`。

**评估协议（影响报数）**：`config.yaml: eval_threshold_on`
- `valid`（默认）：判定阈值在**验证集**上选、固定到测试集——无测试集调参泄漏，**对外投稿用这个**。
- `test`：在测试集上直接搜阈值（旧行为，复现历史 55.29 用），数字偏高、不可对外声称。

---

## 6b. 消融 + 多种子（投稿要用）

每个方法组件都有独立开关（默认全开；`--set X=no` 关掉单个做消融）：
`use_self_loop_fix`(§3.3) / `use_emotion_transition`(§3.3) / `use_necessity`(§3.5) /
`use_pos_prior`(§3.6) / `use_distillation`(§3.7) / `fusion_mode: cond|gated`(§3.8)。

```bash
# 单个消融
python main.py --seed 0 --set use_necessity=no

# 一键跑“完整 + 每个消融 × 种子0/1/2”，每次 25 epoch
bash scripts/run_ablations.sh "0 1 2" 25

# 汇总成 mean±std 表（终端打印 markdown，并写 results/ablation_table.md / .csv）
python scripts/aggregate_results.py results/logs -o results/ablation_table
```

可直接填的主表/消融表模板见 [`results/TABLES.md`](results/TABLES.md)。

---

## 一句话流程

**自测 →（抽样体检）→ 全量标注 →（可选校正）→ 改一行 config → `python main.py`**

投稿额外两步：**默认 `eval_threshold_on: valid` 重跑主表 → `scripts/run_ablations.sh` 出消融表（多种子）**。

---

## 进阶（可选）：自环 SFT 微调老师，提升自环标注质量

7B 老师对**自环**（情绪由本句内容触发）偏弱。可用 gold 自环造一份 LLaMA-Factory
数据微调老师的自环先验。**这是可选支线，不做也能跑完上面 6 步主流程。**

**A. 生成数据（本地、约 1 秒，不调用 LLM）**
```bash
python -m src.build_selfcause_sft \
    --train /root/autodl-tmp/Test/hilo/data/dataset/train.txt \
    --out data/selfcause_sft.json
```
产出 alpaca 格式 JSON：正例=gold 自环（高 `s_dir`/`s_nec`、低 `s_spur`），
负例=gold 里的纯反应句（反过来）。纯正例加 `--no-negatives`。

**B. 注册到 LLaMA-Factory**（加进其 `data/dataset_info.json`）
```json
"selfcause_sft": {
  "file_name": "selfcause_sft.json",
  "formatting": "alpaca",
  "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
}
```

**C. LoRA 微调**
```bash
llamafactory-cli train --stage sft --do_train \
    --model_name_or_path /root/autodl-tmp/Models/Qwen2.5-7B-Instruct \
    --dataset selfcause_sft --dataset_dir <json所在目录> \
    --template qwen --finetuning_type lora --lora_target all \
    --output_dir ckpt/selfcause --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 --learning_rate 1e-4 \
    --num_train_epochs 3 --cutoff_len 4096 --bf16
```

**D. 用微调后的老师做标注**：vLLM 挂上 LoRA 适配器再跑步骤 3 的 `annotate_llm`。

> ⚠️ **泄漏坑**：若微调老师后又去标**同一批 train**，老师会背下 gold → 蒸馏变抄答案，
> valid/test 不会真涨。要么只标没参与微调的数据，要么对 train 做留出/交叉拟合。
> **valid/test 永远不参与微调、不标注。**
