#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end forward/backward integration test for the Method pathway.

BERT and the dataset are not available in this environment, so we patch the
transformers BERT loader with a tiny random encoder and feed synthetic batches
shaped exactly like the data loader's output. This exercises the full wiring:
relation-aware encoder (4 relations), soft gating, perturbation re-encoding,
bounded positional prior, evidence fusion, the two-phase schedule, and the
multi-term loss. Run from the Graph directory:

    python -m tests.test_model_integration
"""

import types
import torch
import torch.nn as nn

import transformers


# ---- tiny fake BERT so model construction needs no checkpoint / network ----
class _FakeBertConfig:
    hidden_size = 16
    num_attention_heads = 4
    attention_probs_dropout_prob = 0.1


class _FakeBert(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embed = nn.Embedding(100, hidden)

    def forward(self, input_ids, attention_mask=None):
        return (self.embed(input_ids),)


def _make_config():
    class Cfg(dict):
        __getattr__ = dict.get
    cfg = Cfg()
    cfg.update(dict(
        bert_path='fake', emo_cat='yes', hid_size=20, dropout=0.1,
        inner_dim=8, loss_weight=2.0, audio_dim=16, video_dim=16, alpha=5,
        speaker_dict={'A': 0, 'B': 1, 'C': 2, 'D': 3},
        device=torch.device('cpu'),
        use_method='yes', warmup_K=1, et_anneal_epochs=2, top_m=8,
        s_pi=10.0, neg_ratio=5.0, pair_pos_weight=2.0, ema_momentum=0.1,
        pos_eta=1.0, alpha1=1.0, alpha2=1.0, beta=1.0, lambda1=1.0,
        lambda2=0.1, gamma=0.01, tau_conflict=0.1, llm_anno_path=None,
    ))
    return cfg


def _make_batch(hidden=16, audio_dim=16, video_dim=16):
    from src.loader import build_mask
    utt_nums = [4, 3]
    B = len(utt_nums)
    max_n = max(utt_nums)
    speakers = [[0, 1, 0, 1], [2, 3, 2]]
    seq_len = 1 + 2 * max_n

    # indices: one packed segment per dialogue; each utterance occupies 2 tokens
    indices = []
    for i, n in enumerate(utt_nums):
        cur = []
        for k in range(n):
            start = 1 + 2 * k
            cur.append((i, start, start + 2))   # (global_id, start, end-exclusive)
        indices.append(cur)

    input_ids = torch.randint(1, 100, (B, seq_len))
    input_masks = torch.ones(B, seq_len, dtype=torch.long)
    utterance_nums = torch.tensor(utt_nums)
    speaker_ids = torch.tensor([s + [0] * (max_n - len(s)) for s in speakers])
    labels = torch.tensor([[1, 2, 0, 3], [4, 0, 5, 0]])   # last col of dialogue 2 is padding
    cause_labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    pairs = torch.tensor([[[1, 0], [3, 2]], [[2, 0], [-100, -100]]])
    pair_nums = torch.tensor([2, 1])
    audio = torch.randn(B, max_n, audio_dim).clamp(-1, 1)
    video = torch.randn(B, max_n, video_dim).clamp(-1, 1)
    gmask, smask, rmask = build_mask(utt_nums, speakers)

    return dict(
        input_ids=input_ids, input_masks=input_masks, indices=indices,
        utterance_nums=utterance_nums, pairs=pairs, pair_nums=pair_nums,
        labels=labels, cause_labels=cause_labels, speaker_ids=speaker_ids,
        doc_ids=[10, 11], video_features=video, audio_features=audio,
        gmasks=gmask, smasks=smask, rmasks=rmask,
    )


def _build_model(cfg):
    import src.model as M
    M.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeBertConfig())
    M.AutoModel = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeBert(_FakeBertConfig.hidden_size))
    model = M.TextClassification(cfg, tokenizer=None)
    return model


def _check_neutral_index():
    """Bug #1: the soft relevance gate's P(non-neutral) must target the *actual*
    neutral class. The 7-way label dict is built by first-seen order, so neutral
    is not necessarily class 0; with the binarized head it is class 0."""
    import torch.nn.functional as F
    print("Checking neutral-class index resolution (bug #1)...")

    # 7-way head: neutral index is data-dependent -> read from label_dict.
    cfg = _make_config()
    cfg['label_dict'] = {'happy': 0, 'angry': 1, 'sad': 2, 'neutral': 3,
                         'surprise': 4, 'fear': 5, 'disgust': 6}
    m = _build_model(cfg)
    assert m.neutral_idx == 3, m.neutral_idx

    # binarized head (emo_cat != 'yes'): neutral is class 0 by construction.
    cfg2 = _make_config(); cfg2['emo_cat'] = 'no'
    m2 = _build_model(cfg2)
    assert m2.neutral_idx == 0, m2.neutral_idx

    # p_emo = 1 - softmax[..., neutral_idx] must equal summed P(non-neutral).
    logits = torch.randn(2, 4, 7)
    p_emo = 1.0 - F.softmax(logits, dim=-1)[..., m.neutral_idx]
    non_neutral = [c for c in range(7) if c != m.neutral_idx]
    p_ref = F.softmax(logits, dim=-1)[..., non_neutral].sum(-1)
    assert torch.allclose(p_emo, p_ref, atol=1e-6)
    print("  [ok] neutral_idx read from label_dict (7-way) and =0 when binarized")


def run():
    print("Running Method pathway integration test (faked BERT)...")
    torch.manual_seed(0)
    _check_neutral_index()
    cfg = _make_config()
    model = _build_model(cfg)
    batch = _make_batch()

    B = batch['input_ids'].shape[0]
    N = batch['labels'].shape[1]

    # ---- warmup epoch (evidence/distillation disabled) ----
    model.train()
    model.global_epoch = 0
    loss, out = model(**batch)
    pair_logits, emo_logits, cause_logits, mask = out
    assert pair_logits.shape == (B, N, N, 2), pair_logits.shape
    assert emo_logits.shape[:2] == (B, N)
    assert cause_logits.shape == (B, N, 2)
    assert torch.isfinite(loss), loss
    loss.backward()
    print(f"  [ok] warmup forward+backward, loss={loss.item():.4f}")

    # ---- post-warmup epoch (evidence + fusion + scheduled sampling active) ----
    model.zero_grad()
    model.global_epoch = 2
    loss2, _ = model(**batch)
    assert torch.isfinite(loss2), loss2
    loss2.backward()
    # the new method heads must receive gradient
    g_score = model.score_head.net[0].weight.grad
    g_fusion = model.evidence_fusion.classifier.weight.grad
    g_pos = model.pos_prior.psi[0].weight.grad
    assert g_score is not None and g_score.abs().sum() > 0
    assert g_fusion is not None and g_fusion.abs().sum() > 0
    assert g_pos is not None and g_pos.abs().sum() > 0
    # 3.3 self-cause fix: the learnable diagonal signature must be live
    g_self = model.self_loop_emb.grad
    assert g_self is not None and g_self.abs().sum() > 0
    print(f"  [ok] active-phase forward+backward, loss={loss2.item():.4f}")
    print("  [ok] score/fusion/positional-prior/self-loop heads all receive gradient")

    # ---- eval mode produces a probability matrix usable by the trainer ----
    model.eval()
    with torch.no_grad():
        _, out_e = model(**batch)
    p = torch.softmax(out_e[0], dim=-1)[..., 1]
    assert ((p >= 0) & (p <= 1)).all()
    print("  [ok] eval pair probability in [0,1]")

    # ---- legacy pathway still runs (use_method='no') ----
    cfg2 = _make_config(); cfg2['use_method'] = 'no'
    legacy = _build_model(cfg2)
    legacy.train()
    lloss, lout = legacy(**batch)
    assert torch.isfinite(lloss)
    lloss.backward()
    print(f"  [ok] legacy pathway forward+backward, loss={lloss.item():.4f}")

    print("Method pathway integration test passed.")


if __name__ == '__main__':
    run()
