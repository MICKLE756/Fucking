# ConvECPE_src — 方法在 ConvECPE (IEMOCAP) 数据集上的适配

将 `../src`（ECF/MECPE 版）的方法章实现（§3.3–§3.10，见 `../英文_改_Method定稿.md` 与 `../paper/main.tex`）迁移到 **ConvECPE** 数据集（[JointEC](https://github.com/Maxwe11y/JointEC) 发布的 `IEMOCAP_emotion_cause_features.pkl`，已放在 `../Dataset/`）。

## 与 ../src 的差异（其余代码保持一致）

| 项 | src (ECF) | ConvECPE_src (IEMOCAP) |
|---|---|---|
| 数据读取 | train/valid/test.txt + npy 特征 | 单个 pkl（`loader.py::read_convecpe`） |
| 情绪类别 | 7 类（neutral 等） | 6 类（happy/sad/neutral/angry/excited/frustrated），label dict 固定 neutral=0 |
| 音频/视觉维度 | 6373 / 4096 | 100 / 512（JointEC 预提取特征） |
| 文本 | BERT 编码原句 | 相同（pkl 中含原句 videoSentence；100 维预提取文本特征弃用） |
| 划分 | 官方 train/valid/test | 官方 trainVid/testVid；从 train 固定随机（seed=13）划出 10% 作验证集用于阈值选择 |
| 原因标注 | pair 列表 | 每条话语最多 3 个原因索引（causeLabels×3），展开为 (emotion, cause) 对 |

方法组件（软候选门控、扰动必要性、位置先验、LLM 蒸馏、互补感知融合、消融开关、验证集选阈值协议）与 `../src` 完全一致；`model.py` 仅把 7 类硬编码改为按 label dict 取类别数。

## 运行

入口为 `../main_conv.py`（与 ECF 的 `../main.py` 并列）。在 `Graph/` 目录下：

```bash
python main_conv.py                       # 完整方法
python main_conv.py --set use_method=no   # 旧基线通路
python main_conv.py --set use_necessity=no  # 消融示例
```

依赖同 `../requirements.txt`；BERT 默认从 HuggingFace 拉取 `bert-base-cased`（可用 `--set bert_path=/path/to/bert` 覆盖）。

## 已知约定/限制

- 与 ECF 管线一致，候选对只枚举 `cause ≤ emotion`（下三角）。ConvECPE 中约 5.8% 金标对的原因出现在情绪话语之后，这部分对召回不可达（各基线同协议）。
- 有原因标注的情绪话语在本数据集中均为非中性，与 §3.4 门控假设一致。
- LLM 蒸馏默认关闭数据侧（`llm_anno_path: null`）；如需 §3.7 离线标注，可参照 `../src/annotate_llm.py` 生成后指向其输出。
