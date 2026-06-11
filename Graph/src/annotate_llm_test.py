"""Quality-check harness for the offline LLM teacher (``annotate_llm``).

Randomly samples a handful of training dialogues (default 50 doc_ids), runs the
same counterfactual annotation used in production, then prints a quality report
so you can eyeball whether the teacher signal is sensible *before* launching the
full annotation run. In particular it separates **self-cause** gold pairs
(cause utterance == emotion utterance, ~half of the gold in this corpus) from
**cross-utterance** gold pairs, so you can verify the dedicated self-cause
prompt branch is actually helping.

Usage (same backend flags as annotate_llm):
    # quick plumbing check, no GPU/LLM:
    python -m src.annotate_llm_test --train data/dataset/train.txt --dry-run

    # real run against a vLLM server, 50 random dialogues:
    python -m src.annotate_llm_test --train data/dataset/train.txt \
        --vllm-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
        --k 5 --num 50 --seed 0 --out data/llm_anno_sample50.pkl

The output pickle has exactly the production schema
``{(doc_id, i, j): [s_nec, s_suf, s_dir, s_spur, rho]}`` so it can be loaded by
``ReasoningAnnotationStore`` if you want to dry-run training on the sample.
"""
import argparse
import os
import pickle
import random
import time
from collections import OrderedDict

from src.annotate_llm import (
    read_dialogues, heuristic_candidates, build_prompt, aggregate,
    StubAnnotator, VLLMAnnotator, QwenAnnotator,
)


def _causal_score(v):
    """A single 'is-cause' summary from the 4 teacher scores: necessity *
    direction * (1 - spuriousness). Used only for the quality report / ranking,
    NOT by training."""
    s_nec, s_suf, s_dir, s_spur = v[0], v[1], v[2], v[3]
    return s_nec * s_dir * (1.0 - s_spur)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def build_annotator(args):
    if args.dry_run:
        return StubAnnotator(k=args.k)
    if args.vllm_url:
        return VLLMAnnotator(args.vllm_url, args.model, k=args.k,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p,
                             api_key=args.api_key)
    if args.qwen_path:
        return QwenAnnotator(args.qwen_path, k=args.k,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature)
    print("[warn] no --vllm-url / --qwen-path; falling back to deterministic stub.")
    return StubAnnotator(k=args.k)


def quality_report(table, sampled, by_doc):
    """Print human-readable quality diagnostics for the sampled annotations."""
    gold = set()                       # (doc_id, i, j) 0-indexed gold pairs
    for d in sampled:
        n = len(d["lines"])
        for (e, c) in d["pairs"]:
            if 1 <= e <= n and 1 <= c <= n:
                gold.add((d["doc_id"], e - 1, c - 1))

    keys = list(table.keys())
    self_keys = [k for k in keys if k[1] == k[2]]
    gold_keys = [k for k in keys if k in gold]
    gold_self = [k for k in gold_keys if k[1] == k[2]]
    gold_cross = [k for k in gold_keys if k[1] != k[2]]
    nongold = [k for k in keys if k not in gold]

    def stat(ks, idx):
        return _mean(table[k][idx] for k in ks)

    print("\n" + "=" * 64)
    print(f"QUALITY REPORT  ({len(sampled)} dialogues, {len(keys)} candidate pairs)")
    print("=" * 64)
    print(f"  self-cause candidates (i==j)      : {len(self_keys)}")
    print(f"  gold pairs covered                : {len(gold_keys)} / {len(gold)}"
          f"   (self {len(gold_self)}, cross {len(gold_cross)})")
    print(f"  mean rho (reliability)            : {stat(keys, 4):.3f}")

    print("\n  --- mean scores: GOLD vs NON-GOLD (higher s_nec / lower s_spur = more causal) ---")
    hdr = f"  {'group':<22}{'n':>6}{'s_nec':>8}{'s_suf':>8}{'s_dir':>8}{'s_spur':>8}{'causal':>8}"
    print(hdr)

    def row(name, ks):
        if not ks:
            print(f"  {name:<22}{0:>6}{'-':>8}{'-':>8}{'-':>8}{'-':>8}{'-':>8}")
            return
        print(f"  {name:<22}{len(ks):>6}{stat(ks,0):>8.3f}{stat(ks,1):>8.3f}"
              f"{stat(ks,2):>8.3f}{stat(ks,3):>8.3f}"
              f"{_mean(_causal_score(table[k]) for k in ks):>8.3f}")

    row("gold (all)", gold_keys)
    row("  gold self-cause", gold_self)
    row("  gold cross-utt", gold_cross)
    row("non-gold", nongold)

    # Ranking quality: per dialogue, how many gold pairs land in the top-|gold|
    # by causal score (precision@|gold| == recall@|gold| here).
    hit, tot = 0, 0
    hit_self, tot_self = 0, 0
    for d in sampled:
        doc = d["doc_id"]
        doc_keys = [k for k in keys if k[0] == doc]
        doc_gold = [k for k in doc_keys if k in gold]
        if not doc_gold:
            continue
        ranked = sorted(doc_keys, key=lambda k: _causal_score(table[k]), reverse=True)
        topk = set(ranked[: len(doc_gold)])
        for k in doc_gold:
            tot += 1
            hit += int(k in topk)
            if k[1] == k[2]:
                tot_self += 1
                hit_self += int(k in topk)
    if tot:
        print(f"\n  gold recovered in top-|gold| by causal score : "
              f"{hit}/{tot} = {hit / tot:.1%}")
    if tot_self:
        print(f"  of which self-cause gold                      : "
              f"{hit_self}/{tot_self} = {hit_self / tot_self:.1%}")

    # A few concrete self-cause gold examples (the case the prompt fix targets).
    print("\n  --- sample self-cause GOLD pairs [s_nec, s_suf, s_dir, s_spur, rho] ---")
    shown = 0
    for k in gold_self:
        doc, i, _ = k
        txt = by_doc[doc]["lines"][i]
        emo = by_doc[doc]["emotions"][i]
        print(f"  doc {doc} utt {i} ({emo}): {[round(x, 3) for x in table[k]]}")
        print(f"      \"{txt[:90]}\"")
        shown += 1
        if shown >= 8:
            break
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/dataset/train.txt")
    ap.add_argument("--out", default="data/llm_anno_sample.pkl")
    ap.add_argument("--num", type=int, default=50, help="how many doc_ids to sample")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the sample")
    ap.add_argument("--qwen-path", default=None)
    ap.add_argument("--vllm-url", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    # The prompt is chain-of-thought (the model reasons in free text over a few
    # steps BEFORE emitting the final flat JSON), so the token budget must be
    # generous: with a small budget the reasoning is truncated before the JSON
    # is ever produced, every sample fails to parse, and *every* pair collapses
    # to the neutral (0.5,0.5,0.5,0.5) fallback with rho=0. 512 leaves ample
    # room for the 3-step reasoning plus the answer.
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dialogues = read_dialogues(args.train)
    if not dialogues:
        raise SystemExit(f"no dialogues parsed from {args.train}")

    rng = random.Random(args.seed)
    num = min(args.num, len(dialogues))
    sampled = rng.sample(dialogues, num)
    sampled.sort(key=lambda d: d["doc_id"])
    by_doc = {d["doc_id"]: d for d in sampled}
    sampled_ids = [d["doc_id"] for d in sampled]
    print(f"Sampled {num} doc_ids (seed={args.seed}): {sampled_ids}")

    cand = {d["doc_id"]: heuristic_candidates(d, args.window) for d in sampled}
    total = sum(len(v) for v in cand.values())
    print(f"Annotating {total} candidate pairs from {num} dialogues ...")

    annotator = build_annotator(args)

    # Build all (key, prompt) pairs up front so we can dispatch them concurrently.
    keys, prompts = [], []
    for doc_id in sampled_ids:
        dlg = by_doc[doc_id]
        n = len(dlg["lines"])
        for (i, j) in cand[doc_id]:
            if not (0 <= i < n and 0 <= j < n):
                continue
            keys.append((doc_id, i, j))
            prompts.append(build_prompt(dlg, i, j))

    table = OrderedDict()
    t0 = time.time()
    # Use the concurrent path when the backend supports it (VLLMAnnotator): it
    # fans the requests out across a thread pool so vLLM can batch them, instead
    # of the ~10x-slower one-pair-at-a-time loop.
    if hasattr(annotator, "batch_annotate"):
        results = annotator.batch_annotate(prompts)
        for key, samples in zip(keys, results):
            table[key] = aggregate(samples, k_expected=args.k)
    else:
        for done, (key, (sys, user)) in enumerate(zip(keys, prompts), 1):
            table[key] = aggregate(annotator.annotate(sys, user), k_expected=args.k)
            if done % 50 == 0 or done == len(keys):
                rate = done / max(1e-6, time.time() - t0)
                print(f"\r  {done}/{len(keys)} pairs ({rate:.1f}/s)", end="", flush=True)
        print()
    print(f"  annotated {len(keys)} pairs in {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(table, f)
    print(f"[done] wrote {len(table)} annotations -> {args.out}")

    # Loudly flag a degenerate run: if (nearly) every pair failed to parse, the
    # pkl is all-neutral with rho=0 and there is NOTHING to compare against gold.
    # This is almost always a backend problem (server unreachable / wrong model)
    # or too small a token budget truncating the chain-of-thought before the JSON.
    n_failed = sum(1 for v in table.values() if v[4] == 0.0)
    if table and n_failed >= 0.9 * len(table):
        print("\n" + "!" * 64)
        print(f"[FATAL] {n_failed}/{len(table)} pairs failed to parse (rho=0): the")
        print("        teacher produced NO usable signal -- the quality report below")
        print("        is meaningless. Likely causes:")
        print("          * vLLM server not reachable / wrong --vllm-url or --model")
        print("          * --max-new-tokens too small: the chain-of-thought is")
        print("            truncated before the final JSON (try --max-new-tokens 512)")
        print("        Send ONE prompt to your server and inspect the raw text before")
        print("        trusting any numbers.")
        print("!" * 64)

    quality_report(table, sampled, by_doc)


if __name__ == "__main__":
    main()
