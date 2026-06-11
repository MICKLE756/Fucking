#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shape / correctness unit tests for the Method-chapter modules in src/method.py.

Run from the Graph directory:
    python -m tests.test_method
These tests exercise the new methodology components on random tensors and do not
require BERT, the dataset, or a GPU.
"""

import torch

from src.method import (
    build_pair_repr, pair_grid,
    BoundedPositionalPrior, AnchoredBaseline, PairScoreHead,
    NecessityEvidence, ReasoningStudent, ReasoningAnnotationStore,
    distillation_loss, EvidenceFusion,
    soft_relevance_gate, presence_weight, balanced_pair_loss,
    sampled_emotion_labels, emotion_transition_matrix,
    conflict_flag, hard_pairs, build_annotation_set,
)
from src.annotate_llm import (
    read_dialogues, heuristic_candidates, build_prompt, parse_scores,
    aggregate, StubAnnotator, build_chat_payload, parse_chat_response,
)

B, N, H = 2, 6, 8
PAIR = 4 * H


def _ok(name):
    print(f"  [ok] {name}")


def test_pair_repr():
    h = torch.randn(B, N, H)
    g = pair_grid(h)
    assert g.shape == (B, N, N, PAIR)
    # [b,i,j] must equal build_pair_repr(h_i, h_j)
    expect = build_pair_repr(h[0, 1], h[0, 3])
    assert torch.allclose(g[0, 1, 3], expect, atol=1e-6)
    _ok("pair_repr / pair_grid layout")


def test_self_cause_repr_fix():
    """3.3 self-cause fix: a learnable signature fills the diagonal's dead
    difference block, leaves cross-pairs and the 4H dimension untouched, and is
    a no-op when the signature is zero (backward compatibility)."""
    h = torch.randn(B, N, H)
    diag = torch.eye(N, dtype=torch.bool)

    # baseline (no self-loop args): the diagonal difference block is dead (all zeros)
    g0 = pair_grid(h)
    assert g0.shape == (B, N, N, PAIR)
    for i in range(N):
        assert torch.allclose(g0[0, i, i, 3 * H:4 * H], torch.zeros(H), atol=1e-6), \
            "diagonal |h_i-h_i| block should be identically zero without the fix"

    # zero signature must be a strict no-op (init-time backward compatibility)
    g_zero = pair_grid(h, self_emb=torch.zeros(H), self_mask=diag)
    assert torch.allclose(g_zero, g0, atol=1e-6)

    # nonzero signature: injected ONLY into the diagonal difference block
    emb = torch.randn(H)
    g1 = pair_grid(h, self_emb=emb, self_mask=diag)
    assert g1.shape == (B, N, N, PAIR)            # dimension preserved (4H)
    for b in range(B):
        for i in range(N):
            # diagonal diff block == |h_i-h_i| (0) + emb == emb
            assert torch.allclose(g1[b, i, i, 3 * H:4 * H], emb, atol=1e-6)
            # the other three blocks of the diagonal are unchanged (h_i, h_i, h_i^2)
            assert torch.allclose(g1[b, i, i, :3 * H], g0[b, i, i, :3 * H], atol=1e-6)
            # product block on the diagonal is still h_i^2
            assert torch.allclose(g1[b, i, i, 2 * H:3 * H], h[b, i] * h[b, i], atol=1e-6)
    # every off-diagonal (cross) pair is bit-for-bit identical to the baseline
    off = ~diag
    assert torch.allclose(g1[:, off], g0[:, off], atol=1e-6)

    # gradients flow back to the signature only through diagonal pairs
    emb_p = torch.zeros(H, requires_grad=True)
    pair_grid(h, self_emb=emb_p, self_mask=diag).sum().backward()
    assert emb_p.grad is not None and torch.allclose(emb_p.grad, torch.full((H,), float(B * N)), atol=1e-4)
    _ok("self-cause representation fix (diagonal signature, dim-preserving)")


def test_positional_prior():
    pp = BoundedPositionalPrior(eta=0.7)
    b = pp(N, torch.device('cpu'))
    assert b.shape == (N, N)
    assert b.abs().max() <= 0.7 + 1e-5            # bounded by eta
    loss = pp.penalty(b)
    assert loss.item() >= 0
    _ok("bounded positional prior in (-eta, eta) + penalty")


def test_anchored_baseline():
    ab = AnchoredBaseline(H, 5, 4)
    ab.update(utt=torch.randn(B, N, H), audio=torch.randn(B, N, 5),
              video=torch.randn(B, N, 4))
    assert ab.baseline_utt().shape == (H,)
    assert ab.baseline_audio().shape == (5,)
    # residual penalty zero at init (delta == 0)
    assert ab.anchor_loss().item() == 0.0
    _ok("anchored baseline EMA + zero-init anchor loss")


def test_score_head_and_necessity():
    f = PairScoreHead(PAIR)
    grid = pair_grid(torch.randn(B, N, H))
    s = f(grid)
    assert s.shape == (B, N, N)
    ne = NecessityEvidence()
    du, da, dv = torch.randn(B, N, N), torch.randn(B, N, N), torch.randn(B, N, N)
    ne.update_stats(du, da, dv)
    z = ne(du, da, dv)
    assert z.shape == (B, N, N, 5)
    # modality weights w^a + w^v sum to 1
    assert torch.allclose(z[..., 3] + z[..., 4], torch.ones(B, N, N), atol=1e-5)
    _ok("score head shape + necessity vector (w^a+w^v=1)")


def test_reasoning_student_and_distill():
    q = ReasoningStudent(PAIR)
    grid = pair_grid(torch.randn(B, N, H)).reshape(-1, PAIR)
    pred = q(grid)
    assert pred.shape == (grid.shape[0], 5)
    assert (pred >= 0).all() and (pred <= 1).all()
    # distillation: half valid
    tgt = torch.rand_like(pred)
    rho = torch.rand(pred.shape[0])
    valid = torch.zeros(pred.shape[0], dtype=torch.bool)
    valid[: pred.shape[0] // 2] = True
    loss = distillation_loss(pred, tgt, rho, valid)
    assert loss.item() >= 0
    # no annotations -> zero loss, still differentiable
    loss0 = distillation_loss(pred, tgt, rho, torch.zeros_like(valid))
    assert loss0.item() == 0.0
    _ok("reasoning student in [0,1] + reliability-weighted distillation")


def test_annotation_store(tmp_path='/tmp/_anno.pkl'):
    import pickle
    table = {(7, 0, 1): [0.9, 0.8, 0.9, 1.0, 0.85]}
    with open(tmp_path, 'wb') as fh:
        pickle.dump(table, fh)
    store = ReasoningAnnotationStore(tmp_path)
    assert store.available()
    t, r, v = store.gather([7, 7], [(7, 0, 1), (7, 2, 3)], torch.device('cpu'))
    assert t.shape == (2, 5) and v.tolist() == [True, False]
    assert abs(r[0].item() - 0.85) < 1e-6
    _ok("offline annotation store lookup + graceful miss")


def test_evidence_fusion():
    pr = pair_grid(torch.randn(B, N, H))
    z_nec = torch.randn(B, N, N, 5)
    z_rea = torch.rand(B, N, N, 5)
    presence = torch.rand(B, N, N)
    s_fact = torch.randn(B, N, N)
    b_pos = torch.rand(N, N).unsqueeze(0).expand(B, N, N)
    for mode in ('gated', 'cond'):
        fusion = EvidenceFusion(PAIR, fusion_mode=mode)
        logit = fusion(pr, z_nec, z_rea, presence, b_pos, s_fact=s_fact)
        assert logit.shape == (B, N, N)
        # necessity evidence must not receive gradient (stop-grad) in either mode
        zg = z_nec.clone().requires_grad_(True)
        fusion(pr, zg, z_rea, presence, b_pos, s_fact=s_fact).sum().backward()
        assert zg.grad is None or torch.count_nonzero(zg.grad) == 0
    # 'cond': low LLM reliability (rho->0) shrinks the teacher's influence so the
    # fused logit moves toward the rho=high counterpart only via the gate, not the
    # raw z_rea. Check the rho-scaling path is actually exercised (finite, no NaN).
    fusion = EvidenceFusion(PAIR, fusion_mode='cond')
    z_lo = z_rea.clone(); z_lo[..., 4] = 0.0
    z_hi = z_rea.clone(); z_hi[..., 4] = 1.0
    out_lo = fusion(pr, z_nec, z_lo, presence, b_pos, s_fact=s_fact)
    out_hi = fusion(pr, z_nec, z_hi, presence, b_pos, s_fact=s_fact)
    assert torch.isfinite(out_lo).all() and torch.isfinite(out_hi).all()
    assert not torch.allclose(out_lo, out_hi)        # rho actually modulates fusion
    _ok("evidence fusion (gated+cond) shape + stop-grad + rho-scaling")


def test_gating_and_presence():
    p_emo = torch.rand(B, N)
    p_cause = torch.rand(B, N)
    pi = soft_relevance_gate(p_emo, p_cause)
    assert pi.shape == (B, N, N)
    valid = torch.triu(torch.ones(N, N), diagonal=0).bool().unsqueeze(0).expand(B, N, N).contiguous()
    g, topM = presence_weight(pi, valid, M=4, s_pi=10.0)
    assert g.shape == (B, N, N)
    # at most M selected per batch element
    for b in range(B):
        assert topM[b].sum() <= 4 + 1   # ties may include the boundary
    _ok("soft relevance gate + top-M presence weighting")


def test_balanced_pair_loss():
    torch.manual_seed(0)
    logits = torch.randn(B, N, N, requires_grad=True)
    gold = torch.zeros(B, N, N)
    gold[0, 2, 1] = 1
    gold[1, 4, 0] = 1
    valid = torch.triu(torch.ones(N, N)).bool().unsqueeze(0).expand(B, N, N).contiguous()
    loss = balanced_pair_loss(logits, gold, valid, neg_ratio=3.0, pos_weight=2.0)
    loss.backward()
    assert loss.item() > 0 and logits.grad is not None
    _ok("balanced pair loss (all-positives + sampled negatives)")


def test_scheduled_sampling_edges():
    gold = torch.randint(0, 7, (B, N))
    ema = torch.randint(0, 7, (B, N))
    s1 = sampled_emotion_labels(gold, ema, epsilon=1.0)
    assert torch.equal(s1, gold)                 # eps=1 -> all gold
    s0 = sampled_emotion_labels(gold, ema, epsilon=0.0)
    assert torch.equal(s0, ema)                  # eps=0 -> all ema
    nums = torch.tensor([N, N - 2])
    et = emotion_transition_matrix(gold, nums)
    assert et.shape == (B, N, N)
    assert not et[0].diagonal().any()            # no self emotion-transition
    assert not et[1, N - 1].any()                # padded utterances excluded
    _ok("scheduled sampling (eps=0/1) + emotion-transition matrix")


def test_annotation_selection():
    p = torch.tensor([[[0.6, 0.9], [0.3, 0.95]]])      # [1,2,2]
    du = torch.tensor([[[0.5, -0.2], [0.5, 0.1]]])
    # conflict: (p>=.5 & du<=0) -> (0,1);  (p<.5 & du>kappa) -> (1,0)
    cf = conflict_flag(p, du, kappa=0.1)
    assert cf[0, 0, 1].item() and cf[0, 1, 0].item()
    # hard pairs: unconfident (max(p,1-p)<tau) OR conflict
    hp = hard_pairs(p, du, tau=0.7, kappa=0.1)
    assert hp[0, 0, 1].item()                          # confident pos but conflicting -> hard
    T = torch.ones(1, 2, 2, dtype=torch.bool)
    S = build_annotation_set(T, p, du, tau=0.7, kappa=0.1, easy_keep=1.0)
    assert S.shape == (1, 2, 2) and S.dtype == torch.bool
    # all hard pairs are always in S
    assert (S | (~hp)).all()
    _ok("hard-pair / Conflict / annotation-set selection")


def test_offline_annotation_pipeline(tmp_path='/tmp/_anno_pipeline.txt'):
    """End-to-end offline teacher pipeline without Qwen: parse dialogues ->
    select candidates -> stub-annotate -> aggregate -> store gather."""
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(
            "1 3\n(2,1),(3,2)\n"
            "1 | A | neutral | Hey what is going on | 00:01\n"
            "2 | B | anger | You broke my mug | 00:02\n"
            "3 | A | sadness | I am sorry | 00:03\n"
        )
    dlgs = read_dialogues(tmp_path)
    assert len(dlgs) == 1 and dlgs[0]['doc_id'] == 1
    cand = heuristic_candidates(dlgs[0], window=12)
    # gold pairs (2,1)->(1,0) and (3,2)->(2,1) must be covered
    assert (1, 0) in cand and (2, 1) in cand
    # parse_scores robustness (counterfactual keys)
    assert parse_scores('noise {"s_nec":0.9,"s_suf":0.1,"s_dir":0.8,"s_spur":0.2} x') \
        == [0.9, 0.1, 0.8, 0.2]
    assert parse_scores('{"s_ec":0.9,"s_ce":0.1,"s_bi":0.8,"g_bi":1.0}') is None  # old keys rejected
    assert parse_scores('not json at all') is None
    ann = StubAnnotator(k=4)
    sys, user = build_prompt(dlgs[0], 1, 0)
    vec = aggregate(ann.annotate(sys, user))
    assert len(vec) == 5 and all(0.0 <= v <= 1.0 for v in vec)
    # build a store-compatible table and gather
    table = {(1, i, j): aggregate(ann.annotate(*build_prompt(dlgs[0], i, j)))
             for (i, j) in cand}
    import pickle as pkl
    with open('/tmp/_anno_pipeline.pkl', 'wb') as f:
        pkl.dump(table, f)
    store = ReasoningAnnotationStore('/tmp/_anno_pipeline.pkl')
    tgt, rho, valid = store.gather([1, 1], [(1, 1, 0), (1, 99, 99)], torch.device('cpu'))
    assert tgt.shape == (2, 5) and valid.tolist() == [True, False]
    _ok("offline LLM annotation pipeline (read->select->stub->store)")


def test_self_cause_prompt_branch(tmp_path='/tmp/_anno_selfcause.txt'):
    """Self-cause (j==i) handling: forced (i,i) candidate + dedicated prompt
    branch + few-shot, both branches keeping the same 4-score schema."""
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(
            "1 3\n(2,2),(3,1)\n"
            "1 | A | neutral | Hey what is going on | 00:01\n"
            "2 | B | sadness | I just found out I lost my job | 00:02\n"
            "3 | A | surprise | Wait, really? | 00:03\n"
        )
    dlg = read_dialogues(tmp_path)[0]

    # (a) forced self candidate (i,i) for every non-neutral emotion utterance,
    #     independent of the window (even window=0 must keep it).
    cand0 = heuristic_candidates(dlg, window=0)
    assert (1, 1) in cand0 and (2, 2) in cand0      # non-neutral utts 2 and 3
    assert (0, 0) not in cand0                       # neutral utt 1 excluded

    # (b) self branch vs cross branch differ in framing but share the schema.
    sys_self, user_self = build_prompt(dlg, 1, 1)
    sys_cross, user_cross = build_prompt(dlg, 1, 0)
    assert 'self-contained' in sys_self and 'self-contained' not in sys_cross
    assert 'SAME utterance' in user_self                       # self framing
    assert 'Candidate cause utterance' in user_cross           # cross framing
    # event-level counterfactual on the self branch vs utterance-level on cross.
    assert 'had NOT happened' in user_self                     # event-level CF
    assert 'had NOT occurred' in user_cross                    # utterance-level CF
    assert 'Think step by step' in user_self and 'Think step by step' in user_cross
    assert 'Examples' in user_self and 'Examples' in user_cross
    for u in (user_self, user_cross):
        assert '"s_nec": <f>' in u and '"s_spur": <f>' in u

    # (c) both branches still parse into a valid 5-d aggregated vector.
    ann = StubAnnotator(k=3)
    for (i, j) in [(1, 1), (1, 0)]:
        vec = aggregate(ann.annotate(*build_prompt(dlg, i, j)))
        assert len(vec) == 5 and all(0.0 <= v <= 1.0 for v in vec)
    _ok("self-cause prompt branch + forced (i,i) candidate + few-shot")


def test_parse_and_reliability_robust():
    """CoT-robust JSON parsing + reliability handling of failed/partial parses."""
    # (1) chain-of-thought: reasoning text contains brace fragments before the
    #     final answer. The greedy first-brace-to-last-brace regex would splice
    #     them together and fail; we must return the LAST valid score object.
    cot = ('Let me think. The schema is {"s_nec": x}.\n'
           'Reasoning: j clearly triggers i, so necessity is high.\n'
           'Final: {"s_nec": 0.9, "s_suf": 0.5, "s_dir": 0.95, "s_spur": 0.1}')
    v = parse_scores(cot)
    assert v == [0.9, 0.5, 0.95, 0.1], v

    # (2) a brace fragment lacking the score keys is skipped, not mis-parsed.
    assert parse_scores('blah {"foo": 1} blah') is None
    assert parse_scores('no json at all') is None

    # (3) all parses failed (empty) -> neutral target with rho == 0 so the pair
    #     is dropped from the rho-weighted distillation loss (was rho==1.0 bug).
    agg_empty = aggregate([])
    assert agg_empty == [0.5, 0.5, 0.5, 0.5, 0.0], agg_empty

    # (4) partial parses -> rho scaled down by the success ratio n/k_expected.
    samples = [[0.8, 0.4, 0.9, 0.1], [0.8, 0.4, 0.9, 0.1]]   # 2 identical, agree
    full = aggregate(samples)                  # no k_expected -> rho == 1.0
    part = aggregate(samples, k_expected=5)    # only 2 of 5 parsed -> 1.0*2/5
    assert abs(full[4] - 1.0) < 1e-9, full
    assert abs(part[4] - 0.4) < 1e-9, part

    # (5) end-to-end: a CoT-braced choice parses; a no-json choice is dropped.
    resp = {"choices": [
        {"message": {"content": cot}},
        {"message": {"content": 'thinking... no answer'}},
    ]}
    vecs = parse_chat_response(resp)
    assert vecs == [[0.9, 0.5, 0.95, 0.1]], vecs
    _ok("CoT-robust parsing + failed/partial-parse reliability (rho)")


def test_gold_anchored_calibration():
    """Bias report + soft gold-anchored calibration of the annotation pkl."""
    from src.calibrate_anno import (
        causal_score, gold_sets, bias_report, calibrate_table,
    )
    # one dialogue: gold pairs (2,2) self-cause and (3,1) cross (1-indexed).
    dlg = {"doc_id": 7, "lines": ["a", "b", "c"],
           "speakers": ["A", "B", "A"],
           "emotions": ["neutral", "sadness", "surprise"],
           "pairs": [(2, 2), (3, 1)]}
    golds = gold_sets([dlg])
    assert golds[7] == {(1, 1), (2, 0)}                 # 0-indexed

    # causal heuristic ordering sanity.
    assert causal_score([0.9, 0.5, 0.9, 0.1]) > causal_score([0.2, 0.2, 0.2, 0.9])

    # table: gold-pos self (underscored), gold-pos cross (ok), gold-neg.
    table = {
        (7, 1, 1): [0.10, 0.10, 0.05, 0.90, 0.80],     # gold self, looks non-causal
        (7, 2, 0): [0.80, 0.50, 0.90, 0.10, 0.90],     # gold cross, already causal
        (7, 2, 1): [0.70, 0.40, 0.80, 0.20, 0.85],     # NON-gold candidate
    }
    rep = bias_report(table, golds, low=0.3, high=0.6)
    assert rep["gold_total"] == 2 and rep["gold_self"] == 1
    assert rep["pos_self"]["n"] == 1 and rep["pos_cross"]["n"] == 1
    assert rep["neg_all"]["n"] == 1
    assert rep["underscored"] == 1 and rep["underscored_self"] == 1   # the self gold pair
    # one gold pair (2,0) present, the other gold (1,1) present -> none missing here
    assert rep["gold_missing_self"] == 0 and rep["gold_missing_cross"] == 0

    # alpha=0 -> identity copy.
    same, st0 = calibrate_table(table, golds, alpha=0.0, beta=0.0)
    assert same == table and st0["n_pos_adjusted"] == 0

    # alpha=0.5, beta=0 -> gold-pos pulled toward causal target, gold-neg untouched.
    new, st = calibrate_table(table, golds, alpha=0.5, beta=0.0)
    assert new[(7, 2, 1)] == table[(7, 2, 1)]                       # non-gold unchanged
    before = causal_score(table[(7, 1, 1)])
    after = causal_score(new[(7, 1, 1)])
    assert after > before                                          # underscored gold lifted
    assert new[(7, 1, 1)][4] >= table[(7, 1, 1)][4]                # rho toward 1
    assert st["n_pos_adjusted"] == 2 and st["n_neg_adjusted"] == 0

    # inject-missing-gold: drop the cross gold from the table, then re-inject.
    tbl2 = {(7, 1, 1): table[(7, 1, 1)]}                           # missing gold (2,0)
    inj, sti = calibrate_table(tbl2, golds, alpha=0.4, inject_missing=True)
    assert (7, 2, 0) in inj and sti["n_injected"] == 1
    assert abs(inj[(7, 2, 0)][4] - 0.4) < 1e-9                     # injected rho = alpha
    _ok("gold-anchored bias report + soft calibration (alpha/beta/inject)")


def test_vllm_payload_and_response():
    """vLLM/OpenAI-compatible request building + multi-sample response parsing."""
    p = build_chat_payload("Qwen/Qwen2.5-7B-Instruct", "sys", "user",
                           k=5, temperature=0.7, top_p=0.9, max_new_tokens=64)
    assert p["n"] == 5 and len(p["messages"]) == 2 and p["max_tokens"] == 64
    resp = {"choices": [
        {"message": {"content": '{"s_nec":0.9,"s_suf":0.1,"s_dir":0.8,"s_spur":0.2}'}},
        {"message": {"content": 'x {"s_nec":0.7,"s_suf":0.2,"s_dir":0.6,"s_spur":0.3} y'}},
        {"message": {"content": 'no json here'}},   # dropped
    ]}
    vecs = parse_chat_response(resp)
    assert len(vecs) == 2
    agg = aggregate(vecs)
    assert len(agg) == 5 and 0.0 <= agg[4] <= 1.0
    assert parse_chat_response({"choices": []}) == []
    _ok("vLLM payload (n=k) + multi-sample chat response parsing")


def main():
    print("Running Method-chapter module tests...")
    test_pair_repr()
    test_self_cause_repr_fix()
    test_positional_prior()
    test_anchored_baseline()
    test_score_head_and_necessity()
    test_reasoning_student_and_distill()
    test_annotation_store()
    test_evidence_fusion()
    test_gating_and_presence()
    test_balanced_pair_loss()
    test_scheduled_sampling_edges()
    test_annotation_selection()
    test_offline_annotation_pipeline()
    test_self_cause_prompt_branch()
    test_parse_and_reliability_robust()
    test_gold_anchored_calibration()
    test_vllm_payload_and_response()
    print("All Method-chapter module tests passed.")


if __name__ == '__main__':
    main()
