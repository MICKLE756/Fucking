#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gold-anchored bias report / soft calibration for the offline LLM annotation pkl.

The LLM teacher signal -- pkl ``{(doc, i, j): [s_nec, s_suf, s_dir, s_spur, rho]}``
produced by ``annotate_llm.py`` -- is a *soft, complementary* signal that is
distilled into the reasoning head ``z^rea``. The gold emotion-cause pairs in
``train.txt`` are the *hard* supervision used directly by the pair-classification
head. They play different roles on purpose.

This tool lets you (a) SEE how far the teacher disagrees with gold and (b)
optionally pull the teacher *softly* toward gold -- without ever hard-overwriting
the pkl and without touching the input file.

  * ``--report`` (the default when ``--alpha``/``--beta`` are 0): prints a bias
    report comparing the pkl against gold, split by self-cause (i==j) vs
    cross-utterance, including how many gold causes the LLM scored as non-causal
    ("underscored") and how many non-gold pairs it scored as causal
    ("overscored"), plus gold pairs missing from the pkl entirely.

  * calibration (``--alpha A`` > 0, optional ``--beta B``): writes a NEW pkl in
    which each gold-positive pair is blended toward a causal target and each
    gold-negative pair (optionally) toward a non-causal target:
        new[0:4] = (1 - a) * llm[0:4] + a * target
        new_rho  = min(1, (1 - a) * llm_rho + a * 1)   # gold-backed -> reliable
    ``alpha = 0`` reproduces the input exactly. The result is written to a new
    file; the input pkl is never modified.

WHY NOT JUST OVERWRITE pkl WITH GOLD?
    Gold is already enforced via the pair loss. Making the teacher an exact copy
    of gold destroys the complementarity between the two paths and leaks the
    training labels into the distillation target, which tends to hurt the held-out
    metrics. Keep ``alpha`` modest (e.g. 0.3) and validate on the valid set.

IMPORTANT: run this ONLY on the *train* annotation pkl. valid/test are never
annotated (annotating them would leak labels), so there is nothing to calibrate.
"""

import argparse
import os
import pickle as pkl
from collections import OrderedDict

from src.annotate_llm import read_dialogues

# Soft targets the gold-positive / gold-negative pairs are pulled toward.
# Positive: necessary, fairly sufficient, clearly directional, not spurious.
# Negative: not necessary/sufficient, weak direction, likely spurious.
T_POS = [0.85, 0.60, 0.90, 0.10]
T_NEG = [0.10, 0.10, 0.20, 0.90]

_KEYS = ("s_nec", "s_suf", "s_dir", "s_spur")


def causal_score(vec):
    """Scalar 'how causal' heuristic in [0,1]: necessity x direction x (1 - spurious).

    Sufficiency is intentionally excluded -- it is consistently conservative and
    would just shrink every score. This mirrors the ranking heuristic used in the
    quality reports.
    """
    s_nec, _s_suf, s_dir, s_spur = vec[0], vec[1], vec[2], vec[3]
    return s_nec * s_dir * (1.0 - s_spur)


def gold_sets(dialogues):
    """doc_id -> set of 0-indexed gold (i=emotion, j=cause) pairs."""
    out = {}
    for d in dialogues:
        n = len(d["lines"])
        s = set()
        for (e, c) in d["pairs"]:                 # gold pairs are 1-indexed
            if 1 <= e <= n and 1 <= c <= n:
                s.add((e - 1, c - 1))
        out[d["doc_id"]] = s
    return out


def _mean(rows, idx):
    return sum(r[idx] for r in rows) / len(rows) if rows else float("nan")


def _group_stats(rows):
    """rows: list of 5-d vectors. Returns dict of per-dim means + causal mean."""
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "s_nec": _mean(rows, 0), "s_suf": _mean(rows, 1),
        "s_dir": _mean(rows, 2), "s_spur": _mean(rows, 3),
        "rho": _mean(rows, 4),
        "causal": sum(causal_score(r) for r in rows) / len(rows),
    }


def bias_report(table, golds, low=0.3, high=0.6):
    """Compare pkl entries against gold; return a structured report dict.

    Splits entries into gold-positive / gold-negative, each further into
    self-cause (i==j) vs cross (i!=j). Also counts gold pairs missing from pkl.
    """
    pos_self, pos_cross, neg_self, neg_cross = [], [], [], []
    for (doc, i, j), vec in table.items():
        is_gold = (i, j) in golds.get(doc, ())
        is_self = (i == j)
        if is_gold:
            (pos_self if is_self else pos_cross).append(vec)
        else:
            (neg_self if is_self else neg_cross).append(vec)

    # gold coverage: which gold pairs are present in the pkl at all.
    present = set(table.keys())
    g_total = g_self = g_cross = 0
    miss_self = miss_cross = 0
    for doc, s in golds.items():
        for (i, j) in s:
            g_total += 1
            self_c = (i == j)
            if self_c:
                g_self += 1
            else:
                g_cross += 1
            if (doc, i, j) not in present:
                if self_c:
                    miss_self += 1
                else:
                    miss_cross += 1

    pos_all = pos_self + pos_cross
    neg_all = neg_self + neg_cross
    under = sum(1 for v in pos_all if causal_score(v) < low)      # gold but looks non-causal
    under_self = sum(1 for v in pos_self if causal_score(v) < low)
    over = sum(1 for v in neg_all if causal_score(v) > high)      # non-gold but looks causal

    return {
        "n_entries": len(table),
        "n_docs": len(golds),
        "gold_total": g_total, "gold_self": g_self, "gold_cross": g_cross,
        "gold_present": g_total - (miss_self + miss_cross),
        "gold_missing_self": miss_self, "gold_missing_cross": miss_cross,
        "pos_self": _group_stats(pos_self), "pos_cross": _group_stats(pos_cross),
        "pos_all": _group_stats(pos_all),
        "neg_self": _group_stats(neg_self), "neg_cross": _group_stats(neg_cross),
        "neg_all": _group_stats(neg_all),
        "underscored": under, "underscored_self": under_self, "n_pos": len(pos_all),
        "overscored": over, "n_neg": len(neg_all),
        "low": low, "high": high,
    }


def _fmt_row(name, g):
    if g.get("n", 0) == 0:
        return f"    {name:<12} n=0"
    return (f"    {name:<12} n={g['n']:<6} "
            f"nec={g['s_nec']:.3f} suf={g['s_suf']:.3f} dir={g['s_dir']:.3f} "
            f"spur={g['s_spur']:.3f} rho={g['rho']:.3f} causal={g['causal']:.3f}")


def print_report(rep):
    print("=" * 70)
    print("  Gold-vs-LLM annotation bias report")
    print("=" * 70)
    print(f"  pkl entries: {rep['n_entries']}    docs with gold: {rep['n_docs']}")
    gt = rep["gold_total"] or 1
    print(f"  gold pairs (train): {rep['gold_total']}  "
          f"self-cause {rep['gold_self']} ({rep['gold_self']/gt*100:.1f}%)  "
          f"cross {rep['gold_cross']}")
    miss = rep["gold_missing_self"] + rep["gold_missing_cross"]
    print(f"  gold present in pkl: {rep['gold_present']}/{rep['gold_total']}  "
          f"(MISSING {miss}: self {rep['gold_missing_self']}, "
          f"cross {rep['gold_missing_cross']})")
    if miss:
        print("    ^ missing gold pairs are never distilled; use --inject-missing-gold "
              "(with --alpha>0) to add them.")
    print("\n  -- gold-POSITIVE entries (LLM scores on true cause pairs) --")
    print(_fmt_row("self-cause", rep["pos_self"]))
    print(_fmt_row("cross", rep["pos_cross"]))
    print(_fmt_row("all", rep["pos_all"]))
    np = rep["n_pos"] or 1
    print(f"    underscored (causal < {rep['low']}): {rep['underscored']}/{rep['n_pos']} "
          f"({rep['underscored']/np*100:.1f}%)  of which self-cause "
          f"{rep['underscored_self']}  <- the LLM's systematic misses")
    print("\n  -- gold-NEGATIVE entries (LLM scores on non-gold candidate pairs) --")
    print(_fmt_row("self-cause", rep["neg_self"]))
    print(_fmt_row("cross", rep["neg_cross"]))
    print(_fmt_row("all", rep["neg_all"]))
    nn = rep["n_neg"] or 1
    print(f"    overscored (causal > {rep['high']}): {rep['overscored']}/{rep['n_neg']} "
          f"({rep['overscored']/nn*100:.1f}%)")
    pc = rep["pos_all"].get("causal", float("nan"))
    nc = rep["neg_all"].get("causal", float("nan"))
    print(f"\n  separation: mean causal  gold-pos {pc:.3f}  vs  gold-neg {nc:.3f}  "
          f"(gap {pc - nc:+.3f})")
    print("=" * 70)


def _clamp(x):
    return min(1.0, max(0.0, x))


def calibrate_table(table, golds, alpha=0.0, beta=0.0, inject_missing=False,
                    t_pos=T_POS, t_neg=T_NEG):
    """Return (new_table, stats). Soft gold-anchored blend; never hard-overwrite.

    gold-positive: new[0:4] = (1-alpha)*llm + alpha*t_pos ; rho -> toward 1 by alpha.
    gold-negative: new[0:4] = (1-beta)*llm + beta*t_neg  ; rho -> toward 1 by beta.
    alpha = beta = 0  =>  identity copy.
    """
    new = OrderedDict()
    n_pos = n_neg = 0
    pos_before = pos_after = 0.0
    for (doc, i, j), vec in table.items():
        is_gold = (i, j) in golds.get(doc, ())
        a = alpha if is_gold else beta
        tgt = t_pos if is_gold else t_neg
        if a <= 0.0:
            new[(doc, i, j)] = list(vec)
            continue
        blended = [_clamp((1.0 - a) * vec[c] + a * tgt[c]) for c in range(4)]
        rho = min(1.0, (1.0 - a) * vec[4] + a * 1.0)
        new[(doc, i, j)] = blended + [rho]
        if is_gold:
            n_pos += 1
            pos_before += causal_score(vec)
            pos_after += causal_score(blended)
        else:
            n_neg += 1

    n_inject = 0
    if inject_missing and alpha > 0.0:
        present = set(table.keys())
        for doc, s in golds.items():
            for (i, j) in s:
                key = (doc, i, j)
                if key in present or key in new:
                    continue
                # No LLM evidence for this pair: seed it at the causal target but
                # trust it only to the extent of alpha (rho = alpha).
                new[key] = list(t_pos) + [float(alpha)]
                n_inject += 1

    stats = {
        "alpha": alpha, "beta": beta, "inject_missing": bool(inject_missing),
        "n_pos_adjusted": n_pos, "n_neg_adjusted": n_neg, "n_injected": n_inject,
        "pos_causal_before": (pos_before / n_pos) if n_pos else float("nan"),
        "pos_causal_after": (pos_after / n_pos) if n_pos else float("nan"),
        "n_out": len(new),
    }
    return new, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True, help="input LLM annotation pkl (train only)")
    ap.add_argument("--train", default="data/dataset/train.txt",
                    help="train.txt providing gold pairs")
    ap.add_argument("--out", default=None,
                    help="output pkl for calibrated table (default: <pkl>_calib.pkl)")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="gold-POSITIVE soft-anchor strength in [0,1] (0 = report only)")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="gold-NEGATIVE soft-anchor strength in [0,1] "
                         "(default 0: leave non-gold pairs untouched -- recommended)")
    ap.add_argument("--inject-missing-gold", action="store_true",
                    help="add gold pairs absent from the pkl, seeded at the causal "
                         "target with rho=alpha (calibrate mode only)")
    ap.add_argument("--low", type=float, default=0.3,
                    help="causal threshold below which a gold pair is 'underscored'")
    ap.add_argument("--high", type=float, default=0.6,
                    help="causal threshold above which a non-gold pair is 'overscored'")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        table = OrderedDict(pkl.load(f))
    dialogues = read_dialogues(args.train)
    golds = gold_sets(dialogues)

    rep = bias_report(table, golds, low=args.low, high=args.high)
    print_report(rep)

    if args.alpha > 0.0 or args.beta > 0.0:
        new, stats = calibrate_table(table, golds, alpha=args.alpha, beta=args.beta,
                                     inject_missing=args.inject_missing_gold)
        out = args.out or (os.path.splitext(args.pkl)[0] + "_calib.pkl")
        if os.path.abspath(out) == os.path.abspath(args.pkl):
            raise SystemExit("[error] --out must differ from --pkl (input is never overwritten)")
        with open(out, "wb") as f:
            pkl.dump(dict(new), f)
        print(f"\n[calibrate] alpha={stats['alpha']} beta={stats['beta']} "
              f"inject_missing={stats['inject_missing']}")
        print(f"  gold-positive adjusted: {stats['n_pos_adjusted']}  "
              f"(mean causal {stats['pos_causal_before']:.3f} -> "
              f"{stats['pos_causal_after']:.3f})")
        print(f"  gold-negative adjusted: {stats['n_neg_adjusted']}")
        print(f"  injected missing gold : {stats['n_injected']}")
        print(f"  wrote {stats['n_out']} entries -> {out}")
        print("  (input pkl left unchanged; validate alpha on the valid set)")
    else:
        print("\n[report-only] no calibration written (pass --alpha > 0 to calibrate).")


if __name__ == "__main__":
    main()
