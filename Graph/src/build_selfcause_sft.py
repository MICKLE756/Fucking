"""Build a self-cause (self-loop, ``j == i``) supervised fine-tuning dataset
for **LLaMA-Factory** from gold labels — a *simple, single-file* generator
(no K-fold cross-fitting).

What it does
------------
Reads ``train.txt`` (gold), finds every gold self-cause utterance (the emotion
is triggered by the event/content described WITHIN the same utterance, i.e. a
gold pair ``(e, e)``), and writes one LLaMA-Factory **alpaca** JSON file:

    {"instruction": <self-cause judging prompt for utterance i>,
     "input": "",
     "output": <short rationale> + <4-score JSON>,
     "system": <self-cause system message>}

``instruction`` / ``system`` are the exact self-cause prompt that
``annotate_llm.build_prompt(d, i, i)`` sends at inference time, so the adapter
is trained on the same prompt distribution it will later see.

Positives = gold self-cause utterances ``(e, e)``               -> HIGH s_dir / s_nec, LOW s_spur
Negatives = gold emotion utterances caused ONLY by others       -> LOW  s_dir / s_nec, HIGH s_spur (pure reaction)

Negatives are included by default so fine-tuning does not collapse to
"everything is self-cause". Use ``--no-negatives`` for positives only.

LEAKAGE WARNING
---------------
Only ever build this from **train**. If you fine-tune the teacher on train gold
and then annotate the SAME train docs with it, the teacher memorizes gold and
the distillation signal becomes a leaked copy of the labels. If you plan to
re-annotate train with the fine-tuned teacher, use a held-out split (or the
cross-fit workflow) so no doc is annotated by an adapter trained on it.
valid/test are NEVER fine-tuned on and NEVER annotated.

Usage
-----
    python -m src.build_selfcause_sft --train /path/to/train.txt \
        --out data/selfcause_sft.json

Then register with LLaMA-Factory by adding the printed ``dataset_info.json``
entry, and fine-tune e.g.::

    llamafactory-cli train --stage sft --do_train \
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
        --dataset selfcause_sft --dataset_dir <dir of the json> \
        --template qwen --finetuning_type lora --lora_target all \
        --output_dir ckpt/selfcause --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 --learning_rate 1e-4 \
        --num_train_epochs 3 --cutoff_len 4096 --bf16
"""
from __future__ import annotations

import argparse
import json
import os

try:
    from .annotate_llm import read_dialogues, build_prompt
except ImportError:  # allow running as a plain script
    from annotate_llm import read_dialogues, build_prompt

# 4-score fine-tuning targets (necessity, sufficiency, direction, spuriousness).
POS_TARGET = (0.85, 0.60, 0.90, 0.10)
NEG_TARGET = (0.12, 0.10, 0.10, 0.88)
_KEYS = ("s_nec", "s_suf", "s_dir", "s_spur")

POS_RATIONALE = ("The emotion is triggered by the event/content described "
                 "within this utterance itself (a self-contained cause).")
NEG_RATIONALE = ("The emotion is a reaction to other utterances; this "
                 "utterance's own content is not the real trigger.")


def selfcause_labels(dialogue):
    """``[(i, label)]`` (0-indexed) for utterances usable as self-cause SFT
    examples. ``label == 1`` iff self-caused in gold; ``label == 0`` iff it is a
    gold emotion utterance caused only by OTHER utterances (pure reaction).
    Utterances with no gold cause at all are skipped (ambiguous)."""
    n = len(dialogue["lines"])
    emotions = dialogue["emotions"]
    gold_emo, gold_self = set(), set()
    for (e, c) in dialogue["pairs"]:            # gold pairs are 1-indexed
        if 1 <= e <= n and 1 <= c <= n:
            gold_emo.add(e - 1)
            if e == c:
                gold_self.add(e - 1)
    out = []
    for i in sorted(gold_emo):
        if emotions[i].lower() == "neutral":    # defensive; gold emo are non-neutral
            continue
        out.append((i, 1 if i in gold_self else 0))
    return out


def _target_json(label, pos, neg):
    vals = pos if label == 1 else neg
    return json.dumps({k: round(float(v), 2) for k, v in zip(_KEYS, vals)})


def build_alpaca_record(dialogue, i, label, pos=POS_TARGET, neg=NEG_TARGET,
                        with_rationale=True):
    """One LLaMA-Factory alpaca record for self-cause utterance ``i``."""
    sys, user = build_prompt(dialogue, i, i)
    rationale = POS_RATIONALE if label == 1 else NEG_RATIONALE
    body = _target_json(label, pos, neg)
    output = f"{rationale}\n{body}" if with_rationale else body
    return {"instruction": user, "input": "", "output": output, "system": sys}


def build_records(dialogues, include_negatives=True,
                  pos=POS_TARGET, neg=NEG_TARGET, with_rationale=True):
    recs = []
    for d in dialogues:
        for (i, label) in selfcause_labels(d):
            if label == 0 and not include_negatives:
                continue
            recs.append(build_alpaca_record(d, i, label, pos, neg, with_rationale))
    return recs


def _dataset_info_entry(file_name):
    return {
        "file_name": os.path.basename(file_name),
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input",
                    "response": "output", "system": "system"},
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="path to train.txt (ONLY train)")
    ap.add_argument("--out", default="data/selfcause_sft.json",
                    help="output LLaMA-Factory alpaca JSON")
    ap.add_argument("--no-negatives", action="store_true",
                    help="emit positives (gold self-cause) only")
    ap.add_argument("--no-rationale", action="store_true",
                    help="JSON-only targets (no leading rationale sentence)")
    ap.add_argument("--pos", type=float, nargs=4, default=list(POS_TARGET),
                    metavar=("NEC", "SUF", "DIR", "SPUR"),
                    help="positive (self-cause) target scores")
    ap.add_argument("--neg", type=float, nargs=4, default=list(NEG_TARGET),
                    metavar=("NEC", "SUF", "DIR", "SPUR"),
                    help="negative (reaction) target scores")
    args = ap.parse_args()

    dialogues = read_dialogues(args.train)
    include_negatives = not args.no_negatives
    recs = build_records(
        dialogues,
        include_negatives=include_negatives,
        pos=tuple(args.pos), neg=tuple(args.neg),
        with_rationale=not args.no_rationale,
    )
    labels = [lab for d in dialogues for (_, lab) in selfcause_labels(d)]
    n_pos = sum(1 for lab in labels if lab == 1)
    n_neg = (sum(1 for lab in labels if lab == 0) if include_negatives else 0)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)

    info = {os.path.splitext(os.path.basename(args.out))[0]: _dataset_info_entry(args.out)}
    print(f"[ok] wrote {len(recs)} alpaca records "
          f"({n_pos} self-cause positives, {n_neg} reaction negatives) -> {args.out}")
    print("Add this to your LLaMA-Factory data/dataset_info.json:")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
