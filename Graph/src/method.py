#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
method.py

Implementation of the finalized Method chapter (英文_改_Method定稿.md) for
multimodal emotion-cause pair extraction (MECPE). The modules here are written
to be self-contained and unit-testable independently of BERT / the dataset, and
are wired into ``model.py`` / ``trainer.py``.

Component map (paper section -> object):
  3.4 Soft candidate gating          -> soft_relevance_gate, select_topM, presence_weight
  3.5 Necessity evidence             -> PairScoreHead, AnchoredBaseline, NecessityEvidence
  3.6 Bounded positional prior       -> BoundedPositionalPrior
  3.7 LLM-guided distillation        -> ReasoningStudent, ReasoningAnnotationStore
  3.8 Evidence fusion and prediction -> EvidenceFusion
  3.9 Training objectives            -> balanced_pair_loss
"""

import os
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pair representation (Eq. h^pair = [h_i; h_j; h_i*h_j; |h_i-h_j|])
# ---------------------------------------------------------------------------
def build_pair_repr(h_i, h_j, self_emb=None, self_mask=None):
    """Compose the candidate-pair representation from two utterance vectors.

    Args:
        h_i: [..., H] emotion-utterance representation.
        h_j: [..., H] cause-utterance representation.
        self_emb: optional [H] learnable self-loop signature. For self-cause
            pairs (i==j) the geometric difference block ``|h_i-h_j|`` is
            identically zero (and ``h_i*h_j`` collapses to ``h_i^2``), so a
            quarter of the pair features carry no information. When provided,
            ``self_emb`` is injected into that otherwise-dead difference block
            at the positions selected by ``self_mask``, giving the diagonal a
            distinct, learnable marker the scoring/fusion heads can exploit.
        self_mask: optional bool tensor broadcastable to the leading dims of
            the pair grid (e.g. [N, N] selecting the diagonal). Only positions
            where it is True receive ``self_emb``.
    Returns:
        [..., 4H] pair representation.
    """
    rep = torch.cat([h_i, h_j, h_i * h_j, (h_i - h_j).abs()], dim=-1)
    if self_emb is not None and self_mask is not None:
        zero = torch.zeros_like(h_i)
        diff_sig = zero + self_emb.to(rep.dtype)            # broadcast [H] -> [...,H]
        add = torch.cat([zero, zero, zero, diff_sig], dim=-1)   # [...,4H]
        rep = rep + add * self_mask.to(rep.dtype).unsqueeze(-1)
    return rep


def pair_grid(h, self_emb=None, self_mask=None):
    """Build the full [B, N, N, 4H] pair grid where index [b, i, j] is pair (u_i, u_j).

    Args:
        h: [B, N, H] utterance representations.
        self_emb, self_mask: optional self-loop signature / position mask,
            forwarded to ``build_pair_repr`` (see its docstring).
    Returns:
        [B, N, N, 4H] pair representations.
    """
    B, N, H = h.shape
    h_i = h.unsqueeze(2).expand(B, N, N, H)   # emotion side varies over dim 1
    h_j = h.unsqueeze(1).expand(B, N, N, H)   # cause side varies over dim 2
    return build_pair_repr(h_i, h_j, self_emb=self_emb, self_mask=self_mask)


# ---------------------------------------------------------------------------
# 3.6 Bounded positional prior:  b^pos = eta * tanh(psi(d_ij)),  d_ij=(i-j)/N
# ---------------------------------------------------------------------------
class BoundedPositionalPrior(nn.Module):
    def __init__(self, eta=1.0, hidden=16):
        super().__init__()
        self.eta = eta
        self.psi = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def distance_matrix(self, N, device):
        idx = torch.arange(N, device=device).float()
        d = (idx.unsqueeze(1) - idx.unsqueeze(0)) / max(N, 1)   # [N, N] in [-1, 1]
        return d

    def forward(self, N, device):
        """Returns the additive logit prior b^pos of shape [N, N] in (-eta, eta)."""
        d = self.distance_matrix(N, device).unsqueeze(-1)        # [N, N, 1]
        b = self.eta * torch.tanh(self.psi(d)).squeeze(-1)       # [N, N]
        return b

    @staticmethod
    def penalty(b_pos, mask=None):
        """L_pos = mean of (b^pos)^2 over the valid region."""
        if mask is not None:
            m = mask.float()
            denom = m.sum().clamp_min(1.0)
            return ((b_pos ** 2) * m).sum() / denom
        return (b_pos ** 2).mean()


# ---------------------------------------------------------------------------
# 3.5 Anchored baselines: b = bar + delta, with EMA-tracked bar (detached) and
#     penalized learnable residual delta.  L_anc = ||delta_utt||^2 + sum_m ||delta_m||^2
# ---------------------------------------------------------------------------
class AnchoredBaseline(nn.Module):
    def __init__(self, utt_dim, audio_dim, video_dim, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        # EMA means (detached, non-trainable buffers)
        self.register_buffer('bar_utt', torch.zeros(utt_dim))
        self.register_buffer('bar_a', torch.zeros(audio_dim))
        self.register_buffer('bar_v', torch.zeros(video_dim))
        self.register_buffer('initialized', torch.zeros(3))
        # learnable, penalized residuals
        self.delta_utt = nn.Parameter(torch.zeros(utt_dim))
        self.delta_a = nn.Parameter(torch.zeros(audio_dim))
        self.delta_v = nn.Parameter(torch.zeros(video_dim))

    @torch.no_grad()
    def _ema(self, buf, batch_mean, slot):
        if self.initialized[slot] == 0:
            buf.copy_(batch_mean)
            self.initialized[slot] = 1.0
        else:
            buf.mul_(1 - self.momentum).add_(self.momentum * batch_mean)

    @torch.no_grad()
    def update(self, utt=None, audio=None, video=None):
        """Update EMA means from a batch. Each arg is [*, dim] or None."""
        if utt is not None:
            self._ema(self.bar_utt, utt.reshape(-1, utt.shape[-1]).mean(0), 0)
        if audio is not None:
            self._ema(self.bar_a, audio.reshape(-1, audio.shape[-1]).mean(0), 1)
        if video is not None:
            self._ema(self.bar_v, video.reshape(-1, video.shape[-1]).mean(0), 2)

    def baseline_utt(self):
        return self.bar_utt.detach() + self.delta_utt

    def baseline_audio(self):
        return self.bar_a.detach() + self.delta_a

    def baseline_video(self):
        return self.bar_v.detach() + self.delta_v

    def anchor_loss(self):
        return (self.delta_utt.pow(2).sum()
                + self.delta_a.pow(2).sum()
                + self.delta_v.pow(2).sum())


# ---------------------------------------------------------------------------
# 3.5 Pair scoring head f: R^{4H} -> R, supervised by L_score = BCE(sigma(s), y)
# ---------------------------------------------------------------------------
class PairScoreHead(nn.Module):
    def __init__(self, pair_dim, hidden=None):
        super().__init__()
        hidden = hidden or pair_dim // 4
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair_repr):
        """pair_repr: [..., 4H] -> scalar score [...]"""
        return self.net(pair_repr).squeeze(-1)


# ---------------------------------------------------------------------------
# 3.5 Necessity evidence assembly with running standardization stats.
# ---------------------------------------------------------------------------
class RunningStd(nn.Module):
    """Tracks running mean/std of a scalar stream for standardization."""

    def __init__(self, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('var', torch.ones(1))
        self.register_buffer('initialized', torch.zeros(1))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1)
        if x.numel() == 0:
            return
        m = x.mean()
        v = x.var(unbiased=False)
        if self.initialized[0] == 0:
            self.mu.copy_(m.view(1))
            self.var.copy_(v.view(1).clamp_min(1e-6))
            self.initialized[0] = 1.0
        else:
            self.mu.mul_(1 - self.momentum).add_(self.momentum * m)
            self.var.mul_(1 - self.momentum).add_(self.momentum * v.clamp_min(1e-6))

    def standardize(self, x, eps=1e-5):
        return (x - self.mu.detach()) / (self.var.detach().sqrt() + eps)


class NecessityEvidence(nn.Module):
    """Builds z^nec = [Δ~^u; Δ~^a; Δ~^v; w^a; w^v] in R^5 (Section 3.5)."""

    def __init__(self, momentum=0.01):
        super().__init__()
        self.std_u = RunningStd(momentum)
        self.std_a = RunningStd(momentum)
        self.std_v = RunningStd(momentum)

    def update_stats(self, du, da, dv):
        self.std_u.update(du)
        self.std_a.update(da)
        self.std_v.update(dv)

    def forward(self, delta_u, delta_a, delta_v):
        """Each delta_*: [...] raw differences s - s^{-*}. Returns z^nec [..., 5]."""
        du = self.std_u.standardize(delta_u)
        da = self.std_a.standardize(delta_a)
        dv = self.std_v.standardize(delta_v)
        # modality weights via softmax over standardized {a, v}
        w = torch.softmax(torch.stack([da, dv], dim=-1), dim=-1)   # [..., 2]
        z = torch.stack([du, da, dv, w[..., 0], w[..., 1]], dim=-1)  # [..., 5]
        return z


# ---------------------------------------------------------------------------
# 3.7 LLM reasoning student q: R^{4H} -> [0,1]^5, distilling the counterfactual
#     teacher vector [s_nec, s_suf, s_dir, s_spur, rho]. Consumes only pair_repr
#     (never the LLM output or z^nec) so it cannot trivially copy the teacher.
# ---------------------------------------------------------------------------
class ReasoningStudent(nn.Module):
    def __init__(self, pair_dim, hidden=None, out_dim=5):
        super().__init__()
        hidden = hidden or pair_dim // 4
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, pair_repr):
        return torch.sigmoid(self.net(pair_repr))   # [..., 5] in [0,1]


class ReasoningAnnotationStore:
    """Offline LLM reasoning-evidence annotations keyed by (doc_id, i, j).

    The store is optional. When the cache file is absent the distillation loss
    is skipped (the student head still produces z^rea for fusion at inference).
    Each entry is a length-5 counterfactual vector
    [s_nec, s_suf, s_dir, s_spur, rho] in [0,1] (see src/annotate_llm.py).
    """

    def __init__(self, path=None):
        self.table = {}
        self.path = path
        if path is not None and os.path.exists(path):
            with open(path, 'rb') as f:
                self.table = pkl.load(f)

    def available(self):
        return len(self.table) > 0

    def lookup(self, doc_id, i, j):
        return self.table.get((int(doc_id), int(i), int(j)))

    def gather(self, doc_ids, ij_list, device):
        """Return (targets [K,5], rho [K], valid_mask [K]) for requested pairs."""
        tgt, rho, valid = [], [], []
        for doc_id, i, j in ij_list:
            v = self.lookup(doc_id, i, j)
            if v is None:
                tgt.append([0.0] * 5)
                rho.append(0.0)
                valid.append(False)
            else:
                tgt.append(list(v))
                rho.append(float(v[4]))
                valid.append(True)
        tgt = torch.tensor(tgt, dtype=torch.float, device=device)
        rho = torch.tensor(rho, dtype=torch.float, device=device)
        valid = torch.tensor(valid, dtype=torch.bool, device=device)
        return tgt, rho, valid


def distillation_loss(student_pred, targets, rho, valid_mask):
    """L_dst = (1/|S|) sum_S rho * sum_c BCE(z_hat_c, z_c)  (Section 3.7).

    Returns a scalar; 0 when no valid annotation is present.
    """
    if valid_mask is None or valid_mask.sum() == 0:
        return student_pred.sum() * 0.0
    pred = student_pred[valid_mask].clamp(1e-6, 1 - 1e-6)
    tgt = targets[valid_mask].clamp(0.0, 1.0)
    r = rho[valid_mask].unsqueeze(-1)
    bce = -(tgt * torch.log(pred) + (1 - tgt) * torch.log(1 - pred))   # [k,5]
    return (r * bce).sum() / valid_mask.sum().clamp_min(1).float()


# ---------------------------------------------------------------------------
# 3.8 Evidence fusion: phi, gate, final logit with bounded positional prior.
# ---------------------------------------------------------------------------
class EvidenceFusion(nn.Module):
    """Section 3.8. Fuse the small model (pair_repr), its perturbation-based
    necessity evidence z^nec, and the LLM-distilled counterfactual evidence
    z^rea into the final pair logit. Two modes:

    'gated'  -- original symmetric gate: g·pair + (1-g)·phi([pair,z_nec,z_rea,a]).
                The gate is unconditioned; complementarity is only implicit.
    'cond'   -- complementarity-aware fusion that operationalizes "big model
                compensates the small model where it is weak; small model leads
                where the big model is too broad / unreliable":
                  * z^rea is reliability-scaled by its self-consistency rho, so an
                    uncertain teacher is automatically down-weighted;
                  * the gate is conditioned on the small model's confidence c_s,
                    on rho, and on the necessity *conflict* between z^nec and
                    z^rea -> unconfident small model leans on the LLM, while an
                    unreliable / conflicting LLM cedes to the small model;
                  * an explicit agreement term boosts the logit when the model's
                    and the LLM's necessity judgements concur (substantive signal
                    rather than a generic blend).
    """

    def __init__(self, pair_dim, nec_dim=5, rea_dim=5, fusion_mode='cond'):
        super().__init__()
        self.fusion_mode = fusion_mode
        if fusion_mode == 'cond':
            # phi sees [pair, z_nec, rho-scaled z_rea, presence, c_s, agreement]
            in_dim = pair_dim + nec_dim + rea_dim + 3
            # gate sees [pair, h_ev, c_s, rho, conflict]
            self.gate = nn.Linear(pair_dim * 2 + 3, pair_dim)
            self.w_agree = nn.Parameter(torch.zeros(1))   # agreement boost (learned, 0-init)
        else:
            in_dim = pair_dim + nec_dim + rea_dim + 1
            self.gate = nn.Linear(pair_dim * 2, pair_dim)
        self.phi = nn.Sequential(
            nn.Linear(in_dim, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, pair_dim),
        )
        self.classifier = nn.Linear(pair_dim, 1)

    def forward(self, pair_repr, z_nec, z_rea, presence, b_pos, s_fact=None):
        """All leading dims broadcast; trailing dims:
        pair_repr [...,4H], z_nec [...,5], z_rea [...,5], presence [...], b_pos [...].
        ``s_fact`` [...] is the small model's factual pair-score logit (confidence
        source for the 'cond' gate); ignored in 'gated' mode.
        Returns logit [...] = w_f^T h_final + b_f + b^pos.
        """
        z_nec = z_nec.detach()                       # necessity evidence is informative-only
        a = presence.unsqueeze(-1)                   # [...,1]

        if self.fusion_mode != 'cond':
            fused_in = torch.cat([pair_repr, z_nec, z_rea, a], dim=-1)
            h_ev = self.phi(fused_in)                # [...,4H]
            g = torch.sigmoid(self.gate(torch.cat([pair_repr, h_ev], dim=-1)))
            h_final = g * pair_repr + (1 - g) * h_ev
            return self.classifier(h_final).squeeze(-1) + b_pos

        # ---- complementarity-aware fusion ----
        rho = z_rea[..., 4:5].detach().clamp(0.0, 1.0)        # LLM reliability proxy
        z_rea_scaled = z_rea * rho                            # unreliable teacher shrinks
        # Fix 4: restore the 5th component to original rho after scaling,
        # so that phi sees the reliability estimate (not rho^2).
        z_rea_scaled = torch.cat([z_rea_scaled[..., :4], rho], dim=-1)
        if s_fact is not None:
            # Confidence from the factual pair-score logit s_{ij} (Section 3.5).
            # This is an intermediate quantity; the final logit also includes
            # positional prior and agreement terms, but s_{ij} serves as a
            # stable prior-confidence proxy that avoids circular dependency
            # with the fusion output (Fix 6).
            c_s = (2.0 * (torch.sigmoid(s_fact) - 0.5)).abs().unsqueeze(-1)
        else:
            c_s = torch.zeros_like(a)
        # necessity agreement: model z^nec[...,0] (standardized Δ~^u) vs LLM s_nec=z_rea[...,0]
        n_model = torch.sigmoid(z_nec[..., 0:1])
        n_llm = z_rea[..., 0:1]
        agree = 1.0 - (n_model - n_llm).abs()                 # [...,1] in [0,1]
        conflict = 1.0 - agree

        fused_in = torch.cat([pair_repr, z_nec, z_rea_scaled, a, c_s, agree], dim=-1)
        h_ev = self.phi(fused_in)
        gate_in = torch.cat([pair_repr, h_ev, c_s, rho, conflict], dim=-1)
        g = torch.sigmoid(self.gate(gate_in))                 # blend small model vs evidence
        h_final = g * pair_repr + (1 - g) * h_ev
        logit = self.classifier(h_final).squeeze(-1) + b_pos
        # Necessity-modulated agreement: only boost when both sources agree
        # the pair IS causal (avg necessity > 0); "both say no" should not
        # push the logit up (Fix 1).
        avg_nec = 0.5 * (n_model.squeeze(-1) + n_llm.squeeze(-1))   # [...] in [0,1]
        logit = logit + self.w_agree * (2.0 * agree.squeeze(-1) - 1.0) * avg_nec
        return logit


# ---------------------------------------------------------------------------
# 3.4 Soft candidate gating
# ---------------------------------------------------------------------------
def soft_relevance_gate(p_emo, p_cause):
    """pi_ij = p^e_i * p^c_j.  p_emo,[B,N]  p_cause,[B,N] -> [B,N,N]."""
    return p_emo.unsqueeze(2) * p_cause.unsqueeze(1)


def presence_weight(pi, valid_mask, M, s_pi):
    """Smooth presence g^ev = sigma(s_pi (pi - pi_M)) for the per-dialogue
    M-th largest relevance pi_M; pairs outside top-M shrink continuously.

    Args:
        pi: [B, N, N] relevance.
        valid_mask: [B, N, N] bool, valid upper-triangular candidate region.
        M: int budget.
        s_pi: temperature.
    Returns:
        g_ev [B, N, N] in [0,1], topM_mask [B, N, N] bool.
    """
    B = pi.shape[0]
    g = torch.zeros_like(pi)
    topM = torch.zeros_like(pi, dtype=torch.bool)
    for b in range(B):
        flat_pi = pi[b][valid_mask[b]]
        if flat_pi.numel() == 0:
            continue
        m = min(M, flat_pi.numel())
        pi_M = torch.topk(flat_pi, m).values.min()
        g[b] = torch.sigmoid(s_pi * (pi[b] - pi_M)) * valid_mask[b].float()
        topM[b] = (pi[b] >= pi_M) & valid_mask[b]
    return g, topM


# ---------------------------------------------------------------------------
# 3.9 Balanced pair loss: all positives + ratio-r sampled negatives, class-weighted.
# ---------------------------------------------------------------------------
def balanced_pair_loss(logits, gold, valid_mask, neg_ratio=5.0, pos_weight=2.0,
                       generator=None):
    """Class-weighted BCE over P+ ∪ N_r within the valid region (Section 3.9).

    Args:
        logits: [B, N, N] pair logits.
        gold:   [B, N, N] {0,1} labels.
        valid_mask: [B, N, N] bool, valid candidate region (upper-tri, padded out).
        neg_ratio: negatives per positive.
        pos_weight: omega, weight on the positive class.
    Returns:
        scalar loss.
    """
    valid = valid_mask.bool()
    pos = valid & (gold > 0.5)
    neg = valid & (gold <= 0.5)

    pos_idx = pos.nonzero(as_tuple=False)
    neg_idx = neg.nonzero(as_tuple=False)
    n_pos = pos_idx.shape[0]
    n_neg = neg_idx.shape[0]

    if n_neg > 0:
        k = int(max(1, round(neg_ratio * max(n_pos, 1))))
        k = min(k, n_neg)
        perm = torch.randperm(n_neg, generator=generator, device=neg_idx.device)[:k]
        neg_idx = neg_idx[perm]

    sel = torch.cat([pos_idx, neg_idx], dim=0) if n_pos > 0 else neg_idx
    if sel.shape[0] == 0:
        return logits.sum() * 0.0

    b, i, j = sel[:, 0], sel[:, 1], sel[:, 2]
    sel_logits = logits[b, i, j]
    sel_gold = gold[b, i, j].float()
    weight = torch.ones_like(sel_gold)
    weight[sel_gold > 0.5] = pos_weight
    return F.binary_cross_entropy_with_logits(sel_logits, sel_gold, weight=weight)


# ---------------------------------------------------------------------------
# 3.5 Necessity calibration loss (Fix 2): direct supervision that pushes
#     Δ^u to be positive for positive pairs and non-positive for negatives.
#     L_cal = (1/|S|) sum 1[y=1] max(0, -Δ^u) + 1[y=0] max(0, Δ^u - κ_cal)
# ---------------------------------------------------------------------------
def necessity_calibration_loss(delta_u, gold, valid_mask, kappa_cal=0.0):
    """Hinge-style calibration of cause necessity against pair labels.

    For positive pairs the cause removal should *decrease* the score
    (Δ^u > 0); for negative pairs it should not increase it beyond a
    small margin κ_cal.

    Args:
        delta_u: [B, N, N] raw cause-necessity difference s - s^{-u}.
        gold:    [B, N, N] {0,1} pair labels.
        valid_mask: [B, N, N] bool.
        kappa_cal: margin for negatives (default 0 → any positive Δ is penalised).
    Returns:
        scalar loss.
    """
    valid = valid_mask.bool()
    pos = valid & (gold > 0.5)
    neg = valid & (gold <= 0.5)
    # positive pairs: penalise Δ^u < 0
    loss_pos = F.relu(-delta_u) * pos.float()
    # negative pairs: penalise Δ^u > κ_cal
    loss_neg = F.relu(delta_u - kappa_cal) * neg.float()
    denom = valid.float().sum().clamp_min(1.0)
    return (loss_pos.sum() + loss_neg.sum()) / denom


# ---------------------------------------------------------------------------
# 3.7 Offline annotation set selection (run once with the frozen warmup model).
# ---------------------------------------------------------------------------
def conflict_flag(p_ij, delta_u, kappa):
    """Conflict_ij (Section 3.7): prediction disagrees with the necessity evidence.

        (p>=.5 and Δ^u<=0)  or  (p<.5 and Δ^u>kappa)
    All inputs are tensors of the same shape; returns a bool tensor.
    """
    pos = (p_ij >= 0.5) & (delta_u <= 0)
    neg = (p_ij < 0.5) & (delta_u > kappa)
    return pos | neg


def hard_pairs(p_ij, delta_u, tau, kappa):
    """P_hard (Section 3.7): unconfident or conflicting pairs (bool tensor)."""
    unconfident = torch.maximum(p_ij, 1 - p_ij) < tau
    return unconfident | conflict_flag(p_ij, delta_u, kappa)


def build_annotation_set(topM_mask, p_ij, delta_u, tau, kappa, easy_keep=0.2,
                         generator=None):
    """S = (T ∩ P_hard) ∪ sample(T \\ P_hard)  (Section 3.7), drawn from the
    same population T used for evidence so the student matches inference.

    Args:
        topM_mask: [B,N,N] bool, the evidence population T.
        p_ij, delta_u: [B,N,N] frozen-model quantities.
        tau, kappa: thresholds.
        easy_keep: fraction of easy pairs in T to sample.
    Returns:
        [B,N,N] bool selection mask S.
    """
    T = topM_mask.bool()
    P_hard = hard_pairs(p_ij, delta_u, tau, kappa)
    hard = T & P_hard
    easy = T & (~P_hard)
    keep = torch.rand(easy.shape, generator=generator, device=easy.device) < easy_keep
    return hard | (easy & keep)


# ---------------------------------------------------------------------------
# 3.3 Emotion-transition edges via scheduled sampling.
# ---------------------------------------------------------------------------
def sampled_emotion_labels(gold_labels, ema_labels, epsilon, generator=None):
    """Scheduled sampling of per-utterance emotion labels (Section 3.3).

    With probability epsilon use gold, else use the detached EMA prediction.

    Args:
        gold_labels: [B, N] long gold emotion ids (may contain pad -100).
        ema_labels:  [B, N] long EMA-predicted ids (detached).
        epsilon: float in [0,1].
    Returns:
        [B, N] long sampled labels.
    """
    use_gold = torch.rand(gold_labels.shape, generator=generator,
                          device=gold_labels.device) < epsilon
    labels = torch.where(use_gold, gold_labels, ema_labels)
    return labels


def emotion_transition_matrix(labels, utterance_nums):
    """Boolean [B, N, N] edge where labels differ between two distinct utterances."""
    B, N = labels.shape
    device = labels.device
    valid = (torch.arange(N, device=device).unsqueeze(0) < utterance_nums.unsqueeze(-1))
    valid2 = valid.unsqueeze(1) & valid.unsqueeze(2)
    diff = labels.unsqueeze(2) != labels.unsqueeze(1)
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    return diff & valid2 & (~eye)
