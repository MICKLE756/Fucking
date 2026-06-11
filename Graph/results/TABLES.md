# 主表 / 消融表模板 (Table templates)

> 所有数字用**多种子均值±方差**（≥3 个种子），评估协议固定为 `eval_threshold_on: valid`（见 `REPRODUCE.md` §4）。
> Pair = 情绪-原因对抽取；Emo / Cause = 情绪句 / 原因句子任务的 F1。

---

## 表 1：主结果（ECF 测试集，vs baseline）

| Method | Pair P | Pair R | Pair F1 | Emo F1 | Cause F1 |
|---|---|---|---|---|---|
| 已发表 baseline（填你对比的工作） |  |  |  |  |  |
| Graph (legacy, `use_method=no`) |  |  |  |  |  |
| **Ours (full method)** |  |  |  |  |  |

填表命令：见 `REPRODUCE.md` §5（baseline 用 `--set use_method=no`，full 用默认）。

---

## 表 2：逐模块消融（在 full 基础上，每次只关一个组件）

行标签与 `--set` 开关一一对应；由 `scripts/aggregate_results.py` 自动生成。

| Config | Pair P | Pair R | Pair F1 | Emo F1 | Cause F1 | ΔPair F1 |
|---|---|---|---|---|---|---|
| **Full**（全开） |  |  |  |  |  | — |
| − Self-loop fix (`use_self_loop_fix=no`, §3.3) |  |  |  |  |  |  |
| − Emotion transition (`use_emotion_transition=no`, §3.3) |  |  |  |  |  |  |
| − Necessity evidence (`use_necessity=no`, §3.5) |  |  |  |  |  |  |
| − Positional prior (`use_pos_prior=no`, §3.6) |  |  |  |  |  |  |
| − Distillation (`use_distillation=no`, §3.7) |  |  |  |  |  |  |
| Fusion → uncond. gate (`fusion_mode=gated`, §3.8) |  |  |  |  |  |  |

> ΔPair F1 = 该消融相对 Full 的 F1 变化（应为负，幅度越大说明该组件越重要）。
> 任何组件若 Δ≈0 或为正，应在论文中讨论或考虑移除（避免“大杂烩”质疑）。

---

## 表 3（可选）：老师质量 ↔ 蒸馏净增益

支撑“蒸了带噪老师仍 net 为正”的论点（见框架评估）。

| 老师标注质量（自环/跨句命中率 或 ρ） | Pair F1 (w/ distill) | Pair F1 (w/o distill) | 净增益 |
|---|---|---|---|
|  |  |  |  |
