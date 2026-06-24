"""意图识别离线评测

针对意图识别「规则层」做离线量化评测：在带标签的样本上跑规则分类器，输出
    - 整体一级意图准确率 / 覆盖率（非弃判占比）/ 覆盖样本上的准确率
    - 各一级意图的 Precision / Recall / F1
    - 混淆矩阵（含「弃判」一列：规则模糊→降级，未直接给出结论）
    - 二级意图准确率（在一级意图判对的样本上）
    - 部分误判样例

规则层是纯本地、零外部依赖、可复现的，因此默认只评测规则层（不需要任何 API Key）。
评测口径与 workflow_nodes.PatentWorkflow 的阈值保持一致：得分 < MIN_SCORE 或
Top1/Top2 分差 < MIN_MARGIN 视为「弃判（abstain）」，对应线上会降级到模型/LLM。

用法：
    cd src
    python eval_intent.py                       # 评测全部训练样本
    python eval_intent.py --min-score 2 --min-margin 1
    python eval_intent.py --errors 30           # 多打印误判样例
"""

import argparse
import json
from pathlib import Path

import config
from intent_recognize_rule_base import IntentRecognizeRuleBase

# 与 workflow_nodes.PatentWorkflow 默认阈值保持一致
DEFAULT_MIN_SCORE = 2.0
DEFAULT_MIN_MARGIN = 1.0

ABSTAIN = "<abstain>"

DATA_FILES = ["patent_intent_train.json", "patent_intent_train_v2.json"]


def load_samples():
    """读取训练样本，返回 [(text, gold_level1, gold_level2)]。"""
    samples = []
    for name in DATA_FILES:
        path = config.BASE_DIR / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            text = item.get("input", "")
            try:
                gold = json.loads(item.get("output", "{}"))
            except json.JSONDecodeError:
                continue
            gold_l1 = gold.get("一级意图", "")
            gold_l2 = gold.get("二级意图", [])
            if isinstance(gold_l2, str):
                gold_l2 = [gold_l2]
            if text and gold_l1:
                samples.append((text, gold_l1, gold_l2))
    return samples


def predict(recognizer, text, min_score, min_margin):
    """返回 (pred_level1, pred_level2_list)。模糊/未命中时 pred_level1 = ABSTAIN。"""
    top = recognizer.top_intent(text)
    if top and top["score"] >= min_score and top["margin"] >= min_margin:
        return top["level1"], top["level2"]
    return ABSTAIN, []


def _pad(text, width):
    """按显示宽度补空格（中文按 2 宽计）。"""
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(0, width - w)


def print_confusion(labels, confusion):
    cols = labels + [ABSTAIN]
    header = _pad("gold\\pred", 16) + "".join(_pad(c, 12) for c in cols)
    print(header)
    print("-" * len(header))
    for g in labels:
        row = _pad(g, 16)
        for p in cols:
            row += _pad(str(confusion[g].get(p, 0)), 12)
        print(row)


def print_metrics(labels, confusion):
    print(f"\n{_pad('intent', 16)}{_pad('precision', 12)}{_pad('recall', 12)}"
          f"{_pad('f1', 12)}{_pad('support', 12)}")
    print("-" * 64)
    macro_f1 = 0.0
    for c in labels:
        tp = confusion[c].get(c, 0)
        fp = sum(confusion[g].get(c, 0) for g in labels if g != c)
        # 漏判：真实为 c 但预测成别的意图或弃判
        fn = sum(v for p, v in confusion[c].items() if p != c)
        support = sum(confusion[c].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        macro_f1 += f1
        print(f"{_pad(c, 16)}{_pad(f'{precision:.3f}', 12)}{_pad(f'{recall:.3f}', 12)}"
              f"{_pad(f'{f1:.3f}', 12)}{_pad(str(support), 12)}")
    print("-" * 64)
    print(f"{_pad('macro avg F1', 16)}{_pad(f'{macro_f1 / len(labels):.3f}', 12)}")


def main():
    parser = argparse.ArgumentParser(description="意图识别规则层离线评测")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--errors", type=int, default=15, help="打印多少条误判样例")
    args = parser.parse_args()

    recognizer = IntentRecognizeRuleBase()
    samples = load_samples()
    if not samples:
        print("未找到带标签样本（patent_intent_train*.json）。")
        return

    labels = list(config.INTENT_INFO.keys())
    confusion = {g: {p: 0 for p in labels + [ABSTAIN]} for g in labels}

    correct = abstain = 0
    l2_correct = l2_total = 0
    errors = []

    for text, gold_l1, gold_l2 in samples:
        pred_l1, pred_l2 = predict(recognizer, text, args.min_score, args.min_margin)
        if gold_l1 not in confusion:
            confusion[gold_l1] = {p: 0 for p in labels + [ABSTAIN]}
        confusion[gold_l1][pred_l1] = confusion[gold_l1].get(pred_l1, 0) + 1

        if pred_l1 == gold_l1:
            correct += 1
            l2_total += 1
            if gold_l2 and pred_l2 and pred_l2[0] == gold_l2[0]:
                l2_correct += 1
        elif pred_l1 == ABSTAIN:
            abstain += 1
            errors.append((text, gold_l1, pred_l1))
        else:
            errors.append((text, gold_l1, pred_l1))

    total = len(samples)
    covered = total - abstain
    print("=" * 64)
    print("意图识别·规则层离线评测")
    print("=" * 64)
    print(f"样本总数: {total}")
    print(f"阈值: min_score={args.min_score}, min_margin={args.min_margin}")
    print(f"一级意图准确率 (全部样本): {correct}/{total} = {correct / total:.1%}")
    print(f"覆盖率 (非弃判占比):       {covered}/{total} = {covered / total:.1%}")
    if covered:
        print(f"覆盖样本上的准确率:        {correct}/{covered} = {correct / covered:.1%}")
    if l2_total:
        print(f"二级意图准确率 (一级判对的样本内): "
              f"{l2_correct}/{l2_total} = {l2_correct / l2_total:.1%}")

    print("\n[混淆矩阵]")
    print_confusion(labels, confusion)
    print("\n[各意图 P/R/F1]")
    print_metrics(labels, confusion)

    if errors:
        print(f"\n[误判/弃判样例 (前 {min(args.errors, len(errors))} 条)]")
        for text, gold_l1, pred_l1 in errors[:args.errors]:
            print(f"  · '{text}'  gold={gold_l1}  pred={pred_l1}")


if __name__ == "__main__":
    main()
