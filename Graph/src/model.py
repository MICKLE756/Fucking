#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from src.layer import EnhancedLSTM , Biaffine, RoEmbedding, SupConLoss, MultiHeadAttention, NewFusionGate, ERGAT
from src.method import (
    pair_grid, build_pair_repr,
    BoundedPositionalPrior, AnchoredBaseline, PairScoreHead,
    NecessityEvidence, ReasoningStudent, ReasoningAnnotationStore,
    distillation_loss, EvidenceFusion,
    soft_relevance_gate, presence_weight, balanced_pair_loss,
    necessity_calibration_loss,
    sampled_emotion_labels,
)

import torch.nn.functional as F

class TextClassification(nn.Module):
    def __init__(self, cfg, tokenizer):
        super(TextClassification, self).__init__()
        self.cfg = cfg 
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
        self.speaker_embedder = nn.Embedding(len(cfg.speaker_dict), bert_config.hidden_size)
        self.tokenizer = tokenizer

        num_classes = 7 if cfg['emo_cat'] == 'yes' else 2
        # Index of the neutral emotion class in ``emotion_logits``. With
        # ``emo_cat != 'yes'`` labels are binarized so neutral is class 0; with
        # the 7-way head the index is data-dependent (the label dict is built by
        # first-seen order), so read it from the label dict rather than assuming 0.
        self.neutral_idx = (
            int(cfg['label_dict']['neutral'])
            if cfg['emo_cat'] == 'yes' and 'label_dict' in cfg and 'neutral' in cfg['label_dict']
            else 0
        )
        num = 2
        self.rope_embedder = RoEmbedding(cfg, bert_config.hidden_size * num)
        self.fusion = NewFusionGate(bert_config.hidden_size * num)

        drop_rate = 0.1

        self.video_linear = nn.Sequential(
            nn.Linear(cfg.video_dim, bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.audio_linear = nn.Sequential(
            nn.Linear(cfg.audio_dim, bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.emotion_linear = nn.Sequential(
            nn.Linear(bert_config.hidden_size * num, cfg.hid_size),
            nn.ReLU(),
            nn.Linear(cfg.hid_size, num_classes)
        )
        self.cause_linear = nn.Linear(bert_config.hidden_size * num, 2)
        self.biaffine = Biaffine(bert_config.hidden_size * num, 2)
        self.contrastive = SupConLoss(temperature=0.1)
        # , bias=(True, False)
        self.dropout = nn.Dropout(cfg['dropout'])

        self.lstm = EnhancedLSTM('drop_connect', bert_config.hidden_size, bert_config.hidden_size, 1, ff_dropout=0.1, recurrent_dropout=0.1, bidirectional=True)

        att_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)

        self.speaker_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)
        self.reply_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)
        self.global_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)

        self.ergat = ERGAT(
            hidden_dim=bert_config.hidden_size * num,
            num_layers=2,
            num_heads=4,
            dropout=drop_rate
        )
        self.graph_gate = NewFusionGate(bert_config.hidden_size * num)

        # ---- Method-chapter components (Sections 3.3-3.9) ----
        self.use_method = str(cfg.get('use_method', 'yes')) == 'yes'
        # ablation switches: default 'yes' == full method (behavior unchanged);
        # set any to 'no' to remove that single component for an ablation study.
        self.use_self_loop_fix = str(cfg.get('use_self_loop_fix', 'yes')) == 'yes'   # 3.3
        self.use_necessity = str(cfg.get('use_necessity', 'yes')) == 'yes'           # 3.5
        self.use_pos_prior = str(cfg.get('use_pos_prior', 'yes')) == 'yes'           # 3.6
        self.use_distillation = str(cfg.get('use_distillation', 'yes')) == 'yes'     # 3.7
        self.use_emotion_transition = str(cfg.get('use_emotion_transition', 'yes')) == 'yes'  # 3.3
        D = bert_config.hidden_size * num          # contextualized utterance dim (H)
        pair_dim = D * 4                            # h^pair dimension (4H)
        self.method_dim = D
        # 3.3 preliminary emotion head: source of predicted labels for emotion-transition edges
        self.pre_emotion_linear = nn.Linear(D, num_classes)
        # 3.5 necessity evidence
        self.score_head = PairScoreHead(pair_dim)
        # 3.3 self-cause representation fix: for self-loop pairs (i==j) the
        # difference block |h_i-h_j| is identically zero, so a quarter of
        # h^pair carries no signal. A learnable self-loop signature is injected
        # into that dead block on the diagonal (consistently in the factual and
        # counterfactual re-scorings, so the necessity delta stays unpolluted).
        # Initialized at zero -> identical to the old behavior at start, then learned.
        self.self_loop_emb = nn.Parameter(torch.zeros(D))
        self.baseline = AnchoredBaseline(D, cfg.audio_dim, cfg.video_dim,
                                         momentum=float(cfg.get('ema_momentum', 0.01)))
        self.necessity = NecessityEvidence(momentum=float(cfg.get('ema_momentum', 0.01)))
        # 3.6 bounded positional prior
        self.pos_prior = BoundedPositionalPrior(eta=float(cfg.get('pos_eta', 1.0)))
        # 3.7 LLM reasoning distillation
        self.reasoning_student = ReasoningStudent(pair_dim)
        self.anno_store = ReasoningAnnotationStore(cfg.get('llm_anno_path', None))
        # 3.8 evidence fusion + final pair classifier
        self.evidence_fusion = EvidenceFusion(
            pair_dim, fusion_mode=cfg.get('fusion_mode', 'cond'))

        # method hyper-parameters
        self.warmup_K = int(cfg.get('warmup_K', 3))
        self.et_anneal_epochs = int(cfg.get('et_anneal_epochs', 5))
        self.topM = int(cfg.get('top_m', 50))
        self.s_pi = float(cfg.get('s_pi', 10.0))
        self.neg_ratio = float(cfg.get('neg_ratio', 5.0))
        self.pair_pos_weight = float(cfg.get('pair_pos_weight', cfg.get('loss_weight', 2.0)))
        self.alpha1 = float(cfg.get('alpha1', 1.0))
        self.alpha2 = float(cfg.get('alpha2', 1.0))
        self.beta = float(cfg.get('beta', 1.0))
        self.lambda1 = float(cfg.get('lambda1', 1.0))
        self.lambda2 = float(cfg.get('lambda2', 0.1))
        self.gamma = float(cfg.get('gamma', 0.01))
        self.tau_conflict = float(cfg.get('tau_conflict', 0.1))
        self.kappa_cal = float(cfg.get('kappa_cal', 0.0))   # necessity calibration margin
        self.global_epoch = 0

        self.apply(self._init_esim_weights)

        self.bert = AutoModel.from_pretrained(cfg.bert_path)
    
    def _init_esim_weights(self, module):
        """
        Initialise the weights of the ESIM model.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # if isinstance(module, nn.Linear):
            # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            # if module.bias is not None:
                # nn.init.zeros_(module.bias)
        # if isinstance(module, nn.Embedding):
            # nn.init.normal_(module.weight, mean=0, std=0.1)
            # nn.init.uniform_(module.weight, -0.1, 0.1)
    
    def get_utt_mask(self, input, utterance_nums, pair_nums, pairs):
        mask = torch.arange(input.shape[1]).unsqueeze(0).to(input.device) < utterance_nums.unsqueeze(-1)
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        triu = torch.flip(torch.flip(torch.triu(torch.ones_like(mask[0])), [1]), [0])
        for i in range(len(utterance_nums)):
            mask[i] = mask[i] * triu
     
        batch_size, seq_len  = input.shape[:2]
        
        gold = input.new_zeros((batch_size, seq_len, seq_len), dtype=torch.long)
        for i in range(len(input)):
            if pair_nums[i] == 0:
                continue
            gold[i, [w[0] for w in pairs[i, :pair_nums[i]]], [w[1] for w in pairs[i, :pair_nums[i]]]] = 1
        return mask, gold
    
    def get_dot_product(self, input, masks, gold_matrix, similarity):
        # input: batch_size, max_utterance_num, hidden_dim
        # utterance_nums: batch_size
        # pairs: batch_size, 2
        product = self.biaffine(input, input).squeeze(-1)
        if len(product.shape) == 3:
            product = product.unsqueeze(-1)
        product = product.transpose(2, 1).transpose(3, 2).contiguous()

        product = product * similarity

        activate_loss = masks.view(-1) == 1
        activate_logits = product.view(-1, 2)[activate_loss]
        activate_gold = gold_matrix.view(-1)[activate_loss]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, self.cfg['loss_weight']]).to(input.device))
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(input.device))
        loss = criterion(activate_logits, activate_gold.long())
        if torch.isnan(loss):
            loss = 0
        return loss, product 
    
    def merge_input(self, input, indices):
        """input: 
        """

        max_utterance_num = max([len(w) for w in indices])

        res = input.new_zeros((len(indices), max_utterance_num, input.shape[-1]))

        for i in range(len(indices)):
            cur_id = indices[i][0][0]
            end_id = indices[i][-1][0]
            cur_lens = 0
            for j in range(cur_id, end_id + 1):
                start = input.new_tensor([w[1] for w in indices[i] if w[0] == j], dtype=torch.long)
                end = input.new_tensor([w[2] - 1 for w in indices[i] if w[0] == j], dtype=torch.long)
                start_rep = torch.gather(input[j], 0, start.unsqueeze(-1).expand(-1, input.shape[-1]))
                end_rep = torch.gather(input[j], 0, end.unsqueeze(-1).expand(-1, input.shape[-1]))

                end = input.new_tensor([w[2] for w in indices[i] if w[0] == j], dtype=torch.long)

                lens = start.shape[0]
                # res[i, cur_lens:cur_lens + lens, :input.shape[-1]] = start_rep
                # res[i, cur_lens:cur_lens + lens, input.shape[-1]:] = end_rep
                res[i, cur_lens:cur_lens + lens] = end_rep + input[j][0].unsqueeze(0)
                cur_lens += lens
        return res
    
    def get_emotion(self, logits, utterance_nums, emotion_labels, emo=True):
        mask = torch.arange(logits.shape[1]).unsqueeze(0).to(logits.device) < utterance_nums.unsqueeze(-1)
        mask = mask.to(logits.device)
        activate_loss = mask.view(-1) == 1
        activate_logits = logits.view(-1, logits.shape[-1])[activate_loss]

        activate_gold = emotion_labels.view(-1)[activate_loss]
        # print(activate_logits.shape)
        if self.cfg.emo_cat == 'yes' and emo:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] + [1.5] * 6).to(logits.device))
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(logits.device))
        loss = criterion(activate_logits, activate_gold.long())
        return loss
    
    def get_kld(self, predtion, speakers, utterance_nums):
        # prediction: batch_size, max_utterance_num, 7
        # kld: batch_size, max_utterance_num
        # first_item: batch_size, max_utterance_num
        # second_item: batch_size, max_utterance_num
        res = []
        # matrix = predtion.new_zeros(predtion.shape[0], predtion.shape[1], predtion.shape[1])
        matrix = torch.eye(predtion.shape[1]).to(predtion.device).unsqueeze(0).repeat(predtion.shape[0], 1, 1)

        for i in range(len(predtion)):
            first_item = predtion[i]
            cur_speaker = speakers[i]
            # print(cur_speaker)
            same_speaker = []
            for j in range(len(cur_speaker)):
                m = 1
                while j - m >= 0 and cur_speaker[j] != cur_speaker[j - m]:
                    m += 1
                same_speaker.append(j - m)
                if m > 1 and j - m != -1:
                    matrix[i, j, j - m: j] = 1
            second_item = predtion[i, same_speaker]

            # 应用softmax函数，使a和b代表概率分布
            a_softmax = F.softmax(first_item, dim=1)
            b_softmax = F.softmax(second_item, dim=1)
            m_x = (a_softmax + b_softmax) / 2

            # kl_div = F.kl_div(a_softmax.log(), b_softmax)
            # kl_div = F.kl_div(b_softmax.log(), a_softmax)
            x = F.kl_div(a_softmax.log(), m_x, reduction='none')
            jsd = 0.5 * F.kl_div(a_softmax.log(), m_x, reduction='none') + 0.5 * F.kl_div(b_softmax.log(), m_x, reduction='none')
            jsd = jsd.sum(-1)
            res.append(jsd)
        res = torch.stack(res)
        res = res.unsqueeze(-1) * matrix

        similarity = res.unsqueeze(-1).repeat(1, 1, 1, 2)
        similarity[..., 0] = 0
        similarity = torch.exp(similarity * self.cfg.alpha)
        # for w in res:
            # for kk in w:
                # print([round(w * 100, 2) for w in kk.tolist()])
        return similarity 
    
    def build_attention(self, sequence_outputs, gmasks=None, smasks=None, rmasks=None):
        """
        sequence_outputs: batch_size, seq_len, hidden_size
        speaker_matrix: batch_size, num, num 
        head_matrix: batch_size, num, num 
        """
        # speaker_masks = smasks.bool().unsqueeze(1)
        # reply_masks = rmasks.bool().unsqueeze(1)
        # global_masks = gmasks.bool().unsqueeze(1)


        rep = self.reply_attention(sequence_outputs, sequence_outputs, sequence_outputs, rmasks)[0]
        thr = self.global_attention(sequence_outputs, sequence_outputs, sequence_outputs, gmasks)[0]
        sp = self.speaker_attention(sequence_outputs, sequence_outputs, sequence_outputs, smasks)[0]
        r = torch.stack((rep, thr, sp), 0)
        r = torch.max(r, 0)[0]

        length = sequence_outputs.shape[1] // 4

        return r[:, : length]


    def _encode_pregraph(self, text, speaker_emb, audio, video, utterance_nums, gmasks, smasks, rmasks):
        """Pre-graph encoder: modality+speaker fusion -> LSTM -> relation attention -> gate.
        Returns the contextualized representation fed to the relation graph."""
        x = text + speaker_emb + audio + video
        x = self.lstm(x, None, utterance_nums.cpu())
        tt = torch.cat((text, text), dim=-1)
        aa = torch.cat((audio, audio), dim=-1)
        vv = torch.cat((video, video), dim=-1)
        seq = torch.cat((x, tt, aa, vv), 1)
        out = self.build_attention(seq, gmasks, smasks, rmasks)
        return self.fusion(x, out)

    def _encode_graph(self, x_fused, speaker_ids, utterance_nums, emotion_labels):
        """Relation-aware graph propagation (Section 3.3). Returns \\tilde h."""
        graph_out = self.ergat(x_fused, speaker_ids, utterance_nums, emotion_labels)
        return self.graph_gate(x_fused, graph_out)

    def _self_loop_args(self, N, device):
        """Self-cause (diagonal) signature + position mask for build_pair_repr.

        Returns (self_loop_emb, diag_mask) when the method path is active, else
        (None, None) so the pair representation is bit-for-bit the original."""
        if not self.use_method or not self.use_self_loop_fix:
            return None, None
        diag = torch.eye(N, device=device, dtype=torch.bool)   # [N, N]
        return self.self_loop_emb, diag

    def _necessity_evidence(self, h, text, speaker_emb, audio, video,
                            audio_features, video_features, speaker_ids, et_labels,
                            utterance_nums, gmasks, smasks, rmasks, s_fact):
        """Perturbation-based necessity evidence (Section 3.5).
        Cause necessity is representation-level (re-score f); modality necessity for
        {a, v} is input-level via two shared whole-conversation re-encodings, with
        an additional pair-local modality term (Fix 3) that replaces only the cause
        side, making it parallel to the cause-necessity computation.
        Returns (z_nec, anc_loss, delta_u)."""
        B, N, D = h.shape
        sc_emb, sc_mask = self._self_loop_args(N, h.device)

        # --- cause necessity (representation-level): replace \tilde h_j by anchored baseline ---
        b_utt = self.baseline.baseline_utt().view(1, 1, 1, D)
        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j_base = b_utt.expand(B, N, N, D)
        s_minus_u = self.score_head(build_pair_repr(h_i, h_j_base, self_emb=sc_emb, self_mask=sc_mask))
        delta_u = s_fact - s_minus_u

        # --- modality necessity (input-level): re-encode with baseline at the input ---
        audio_base = self.audio_linear(
            self.baseline.baseline_audio().view(1, 1, -1).expand(B, N, -1))
        video_base = self.video_linear(
            self.baseline.baseline_video().view(1, 1, -1).expand(B, N, -1))

        x_a = self._encode_pregraph(text, speaker_emb, audio_base, video, utterance_nums, gmasks, smasks, rmasks)
        h_a = self._encode_graph(x_a, speaker_ids, utterance_nums, et_labels)

        x_v = self._encode_pregraph(text, speaker_emb, audio, video_base, utterance_nums, gmasks, smasks, rmasks)
        h_v = self._encode_graph(x_v, speaker_ids, utterance_nums, et_labels)

        # Fix 3: pair-local modality perturbation — replace only the cause side
        # h_j^{-m,local} from the global re-encoding, keep h_i from the original.
        # This parallels the cause-necessity computation (which also only replaces h_j)
        # and measures "how much does modality m in the cause utterance matter for this pair".
        delta_a_local = s_fact - self.score_head(build_pair_repr(
            h_i, h_a.unsqueeze(2).expand(B, N, N, D), self_emb=sc_emb, self_mask=sc_mask))
        delta_v_local = s_fact - self.score_head(build_pair_repr(
            h_i, h_v.unsqueeze(2).expand(B, N, N, D), self_emb=sc_emb, self_mask=sc_mask))

        if self.training:
            self.necessity.update_stats(delta_u.detach(), delta_a_local.detach(), delta_v_local.detach())
        z_nec = self.necessity(delta_u, delta_a_local, delta_v_local)     # [B,N,N,5]
        return z_nec, self.baseline.anchor_loss(), delta_u

    def _distillation_loss(self, z_rea, doc_ids, topM):
        """Reliability-weighted grouped BCE over offline-annotated pairs (Section 3.7)."""
        device = z_rea.device
        ij_list, idx = [], []
        B = z_rea.shape[0]
        for b in range(B):
            for i, j in topM[b].nonzero(as_tuple=False).tolist():
                ij_list.append((int(doc_ids[b]), int(i), int(j)))
                idx.append((b, i, j))
        if not ij_list:
            return z_rea.sum() * 0.0
        tgt, rho, valid = self.anno_store.gather([t[0] for t in ij_list], ij_list, device)
        preds = torch.stack([z_rea[b, i, j] for (b, i, j) in idx], dim=0)
        return distillation_loss(preds, tgt, rho, valid)

    def forward(self, **kwargs):
        input_ids, input_masks, utterance_nums = [kwargs[w] for w in 'input_ids input_masks utterance_nums'.split()]
        pairs, pair_nums, labels, indices = [kwargs[w] for w in 'pairs pair_nums labels indices'.split()]
        cause_labels, speaker_ids = [kwargs[w] for w in ['cause_labels', 'speaker_ids']]
        audio_features, video_features = [kwargs[w] for w in ['audio_features', 'video_features']]
        gmasks, smasks, rmasks = [kwargs[w] for w in ['gmasks', 'smasks', 'rmasks']]

        bert_out = self.bert(input_ids, attention_mask=input_masks)[0]
        speaker_emb = self.speaker_embedder(speaker_ids)
        text = self.merge_input(bert_out, indices)
        audio = self.audio_linear(audio_features)
        video = self.video_linear(video_features)

        if not self.use_method:
            # ---- legacy Graph(55.29) pathway ----
            input = self._encode_pregraph(text, speaker_emb, audio, video, utterance_nums, gmasks, smasks, rmasks)
            graph_out = self.ergat(input, speaker_ids, utterance_nums)
            input = self.graph_gate(input, graph_out)
            emotion_logits = self.emotion_linear(input)
            emo_loss = self.get_emotion(emotion_logits, utterance_nums, labels, emo=True)
            cause_logits = self.cause_linear(input)
            cause_loss = self.get_emotion(cause_logits, utterance_nums, cause_labels, emo=False)
            ecp_mask, gold_matrix = self.get_utt_mask(input, utterance_nums, pair_nums, pairs)
            similarity = self.get_kld(emotion_logits, speaker_ids, utterance_nums)
            ecp_loss, ecp_logits = self.get_dot_product(input, ecp_mask, gold_matrix, similarity)
            rop_loss, rop_logits = self.rope_embedder.classify_matrix(input, gold_matrix, ecp_mask, similarity)
            loss = rop_loss + emo_loss + cause_loss + ecp_loss
            return loss, (rop_logits + ecp_logits, emotion_logits, cause_logits, ecp_mask)

        # ================= Method-chapter pathway (Sections 3.3-3.9) =================
        evidence_active = self.global_epoch >= self.warmup_K

        # --- 3.3 pre-graph encode + preliminary emotion (emotion-transition edge source) ---
        x_fused = self._encode_pregraph(text, speaker_emb, audio, video, utterance_nums, gmasks, smasks, rmasks)
        pre_emo_logits = self.pre_emotion_linear(x_fused)
        pred_labels = pre_emo_logits.argmax(-1).detach()

        # scheduled-sampled emotion-transition labels (gold during warmup, anneal toward predicted)
        if self.training:
            if evidence_active:
                eps = max(0.0, 1.0 - (self.global_epoch - self.warmup_K) / max(1, self.et_anneal_epochs))
            else:
                eps = 1.0
            gold_for_et = torch.where(labels >= 0, labels, pred_labels)
            et_labels = sampled_emotion_labels(gold_for_et, pred_labels, eps)
        else:
            et_labels = pred_labels

        # --- 3.3 relation-aware graph encode (4 relations) ---
        # Fix 7: emotion-transition edges are activated only after warmup,
        # consistent with the paper statement. During warmup, pass None so
        # the graph uses only structural relations (self-loop, same-speaker,
        # reply/adjacent).
        et_for_graph = et_labels if (evidence_active and self.use_emotion_transition) else None
        h = self._encode_graph(x_fused, speaker_ids, utterance_nums, et_for_graph)

        # auxiliary heads
        emotion_logits = self.emotion_linear(h)
        cause_logits = self.cause_linear(h)
        emo_loss = self.get_emotion(emotion_logits, utterance_nums, labels, emo=True)
        cause_loss = self.get_emotion(cause_logits, utterance_nums, cause_labels, emo=False)
        pre_emo_loss = self.get_emotion(pre_emo_logits, utterance_nums, labels, emo=True)

        ecp_mask, gold_matrix = self.get_utt_mask(h, utterance_nums, pair_nums, pairs)
        valid_mask = ecp_mask.bool()

        # --- 3.5 pair representation + factual score head ---
        sc_emb, sc_mask = self._self_loop_args(h.shape[1], h.device)
        pair = pair_grid(h, self_emb=sc_emb, self_mask=sc_mask)        # [B,N,N,4H]
        s_fact = self.score_head(pair)                                 # [B,N,N]
        score_loss = balanced_pair_loss(s_fact, gold_matrix, valid_mask,
                                        self.neg_ratio, self.pair_pos_weight)

        # --- 3.4 soft relevance gate + presence weighting ---
        p_emo = 1.0 - F.softmax(emotion_logits, dim=-1)[..., self.neutral_idx]  # P(non-neutral) [B,N]
        p_cause = F.softmax(cause_logits, dim=-1)[..., 1]              # [B,N]
        pi = soft_relevance_gate(p_emo, p_cause)                       # [B,N,N]
        g_ev, topM = presence_weight(pi, valid_mask, self.topM, self.s_pi)

        # --- 3.5 necessity evidence (only after warmup; ablatable) ---
        if evidence_active and self.use_necessity:
            z_nec, anc_loss, delta_u = self._necessity_evidence(
                h, text, speaker_emb, audio, video, audio_features, video_features,
                speaker_ids, et_labels, utterance_nums, gmasks, smasks, rmasks, s_fact)
            if self.training:
                self.baseline.update(utt=h.detach(),
                                     audio=audio_features.detach(),
                                     video=video_features.detach())
            # Fix 2: necessity calibration loss
            cal_loss = necessity_calibration_loss(delta_u, gold_matrix, valid_mask,
                                                   self.kappa_cal)
        else:
            z_nec = pi.new_zeros(*pi.shape, 5)
            anc_loss = self.baseline.anchor_loss() * 0.0
            cal_loss = s_fact.sum() * 0.0

        # --- 3.7 LLM reasoning student ---
        z_rea = self.reasoning_student(pair)                           # [B,N,N,5]
        # Fix 5: during warmup, zero out the presence weight so evidence does
        # NOT enter fusion, but keep z_rea non-zero so the student head
        # receives gradients from the distillation loss and is pre-trained.
        if not evidence_active:
            g_ev = torch.zeros_like(g_ev)
        # ablation: drop the distilled reasoning evidence from fusion (the student
        # head is still built but contributes nothing to the final logit).
        z_rea_fused = z_rea if self.use_distillation else torch.zeros_like(z_rea)

        # --- 3.6 bounded positional prior (ablatable) ---
        N = h.shape[1]
        if self.use_pos_prior:
            b_pos = self.pos_prior(N, h.device).unsqueeze(0).expand(pi.shape[0], N, N)
            pos_loss = self.pos_prior.penalty(b_pos, valid_mask)
        else:
            b_pos = pi.new_zeros(pi.shape[0], N, N)
            pos_loss = s_fact.sum() * 0.0

        # --- 3.8 evidence fusion + final pair logit ---
        final_logit = self.evidence_fusion(pair, z_nec, z_rea_fused, g_ev, b_pos,
                                           s_fact=s_fact.detach())            # [B,N,N]
        pair_loss = balanced_pair_loss(final_logit, gold_matrix, valid_mask,
                                       self.neg_ratio, self.pair_pos_weight)

        # --- 3.7 distillation (Fix 5: also active during warmup if annotations exist) ---
        if self.use_distillation and self.anno_store.available():
            dst_loss = self._distillation_loss(z_rea, kwargs.get('doc_ids'), topM)
        else:
            dst_loss = final_logit.sum() * 0.0

        # --- 3.9 total objective ---
        L_cls = pair_loss + self.alpha1 * emo_loss + self.alpha2 * cause_loss + pre_emo_loss
        loss = (L_cls
                + self.beta * score_loss
                + self.lambda1 * dst_loss
                + self.lambda2 * pos_loss
                + self.gamma * anc_loss
                + self.gamma * cal_loss)

        # 2-class pair logits for trainer compatibility: softmax(.,-1)[...,1] == sigma(final_logit)
        pair_logits_2c = torch.stack([torch.zeros_like(final_logit), final_logit], dim=-1)
        return loss, (pair_logits_2c, emotion_logits, cause_logits, ecp_mask)