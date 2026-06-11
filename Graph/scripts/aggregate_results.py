#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate ablation logs produced by scripts/run_ablations.sh into a
mean +/- std table (markdown + CSV).

Each log file is named <config>_seed<seed>.log and contains the trainer's
final test report, whose last block looks like:

    Pair Pre. 53.1234   Rec. 57.8910   F1 55.4000   (th=0.120)
    TP 1234  Pred. 2300  Gold. 2100
    Emo: Pre. ...  Rec. ...  F1 70.1234
    Cause: Pre. ...  Rec. ...  F1 68.5678

We parse the LAST occurrence of each metric line (the test split printed by
final_evaluate, after all per-epoch validation prints).

Usage:
    python scripts/aggregate_results.py results/logs [-o results/ablation_table]
"""
import argparse
import csv
import os
import re
import statistics
from collections import defaultdict

PAIR_RE = re.compile(r"Pair Pre\.\s*([\d.]+)\s*Rec\.\s*([\d.]+)\s*F1\s*([\d.]+)")
EMO_RE = re.compile(r"Emo:\s*Pre\.\s*[\d.]+\s*Rec\.\s*[\d.]+\s*F1\s*([\d.]+)")
CAUSE_RE = re.compile(r"Cause:\s*Pre\.\s*[\d.]+\s*Rec\.\s*[\d.]+\s*F1\s*([\d.]+)")
NAME_RE = re.compile(r"^(?P<config>.+)_seed(?P<seed>\d+)\.log$")

# preferred display order; anything else is appended alphabetically
ORDER = ["full", "no_self_loop_fix", "no_necessity", "no_pos_prior",
         "no_distillation", "no_emotion_transition", "fusion_gated"]


def _last(regex, text, n=1):
    m = regex.findall(text)
    if not m:
        return None
    last = m[-1]
    return float(last) if n == 1 else tuple(float(x) for x in last)


def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    pair = _last(PAIR_RE, text, n=3)
    if pair is None:
        return None
    p, r, f1 = pair
    return {
        "pair_p": p, "pair_r": r, "pair_f1": f1,
        "emo_f1": _last(EMO_RE, text),
        "cause_f1": _last(CAUSE_RE, text),
    }


def fmt(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f}±{statistics.pstdev(vals):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", help="directory with <config>_seed<seed>.log files")
    ap.add_argument("-o", "--out", default=None,
                    help="output prefix; writes <prefix>.md and <prefix>.csv")
    args = ap.parse_args()

    runs = defaultdict(lambda: defaultdict(list))  # config -> metric -> [values]
    n_seeds = defaultdict(set)
    for fn in sorted(os.listdir(args.logdir)):
        m = NAME_RE.match(fn)
        if not m:
            continue
        res = parse_log(os.path.join(args.logdir, fn))
        if res is None:
            print(f"[warn] no test metrics found in {fn} (run may have crashed)")
            continue
        cfg = m.group("config")
        n_seeds[cfg].add(m.group("seed"))
        for k, v in res.items():
            runs[cfg][k].append(v)

    if not runs:
        print("No parseable logs found.")
        return

    configs = [c for c in ORDER if c in runs] + \
              sorted(c for c in runs if c not in ORDER)
    cols = ["pair_p", "pair_r", "pair_f1", "emo_f1", "cause_f1"]
    headers = ["Config", "#seeds", "Pair P", "Pair R", "Pair F1", "Emo F1", "Cause F1"]

    rows = []
    for cfg in configs:
        row = [cfg, str(len(n_seeds[cfg]))] + [fmt(runs[cfg][c]) for c in cols]
        rows.append(row)

    # ---- markdown ----
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    table = "\n".join(md)
    print(table)

    if args.out:
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(table + "\n")
        with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        print(f"\nwrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
