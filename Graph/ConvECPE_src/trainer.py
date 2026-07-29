#!/use/bin/env python
#!/usr/bin/env python

"""
Name: trainer.py
"""

import os

import torch
import torch.nn.functional as F

import numpy as np
import torch.nn as nn
# import wandb

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

class MyTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        
        # 用于记录最佳结果
        self.scores = []
        self.lines = []
        self.re_init()

    def train(self):
        best_score, best_iter = 0, -1
        self.best_val_threshold = 0.05

        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            
            # 训练和评估
            self.train_step()
            score, (res, _) = self.evaluate_step()
            print(f"\n{res}\n")
            # decision threshold tuned on the *validation* split this epoch
            val_threshold = self.best_threshold

            # 重置统计数据并记录结果
            self.re_init()
            self.add_instance(score, res)

            # 保存最佳模型
            if score > best_score:
                if best_iter > -1:
                    os.remove(self.save_name.format(best_iter))
                best_score, best_iter = score, epoch
                # remember the validation-selected threshold of the best epoch,
                # so the test split is scored without tuning on test itself.
                self.best_val_threshold = val_threshold
                
                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                
                torch.save(
                    {
                        'epoch': epoch,
                        'model': self.model.cpu().state_dict(),
                        'best_score': best_score
                    },
                    self.save_name.format(epoch)
                )
                self.model.to(self.config.device)
            
            # 早停
            elif epoch - best_iter > self.config.patience:
                print(f"Not upgrade for {self.config.patience} steps, early stopping...")
                break
                
        # 最终评估
        score, res = self.final_evaluate(best_iter)
        self.final_score, self.final_res = score, res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader)
        losses = []

        for data in train_data:
            loss, _ = self.model(**data)
            losses.append(loss.item())
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.config.optimizer.step()
            self.model.zero_grad()

            train_data.set_description(
                f"Epoch {self.global_epoch}, loss:{np.mean(losses):.4f}"
            )

    def evaluate_step(self, dataLoader=None, fixed_threshold=None):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        
        for data in dataLoader:
            with torch.no_grad():
                loss, output = self.model(**data)
                self.add_output(data, output)
        
        return self.report_score(fixed_threshold=fixed_threshold)

    def final_evaluate(self, epoch=0):
        checkpoint = torch.load(self.save_name.format(epoch), map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.config.device)
        self.model.eval()

        # Score the test split. Default protocol applies the threshold tuned on
        # the validation split (no in-sample tuning on test); set
        # eval_threshold_on='test' to reproduce the legacy in-sample sweep.
        mode = str(self.config.get('eval_threshold_on', 'valid'))
        fixed = getattr(self, 'best_val_threshold', None) if mode == 'valid' else None
        self.re_init()
        score, res = self.evaluate_step(self.test_loader, fixed_threshold=fixed)
        print(res[0])
        return score, res

    def re_init(self):
        self.preds = defaultdict(list)
        self.golds = defaultdict(list)
        # candidate pairs with joint scores: list of (doc_id, w, z, joint_score)
        self.cand_pairs = []
        self.keys = ['default']

    def add_instance(self, score, res):
        self.scores.append(score)
        self.lines.append(res)

    def add_output(self, data, output):
        ecp_predictions, emo_predictions, cause_predictions, masks = output

        # === Soft probability joint scoring ===
        # ecp_predictions: [B, N, N, 2]; emo_predictions: [B, N, C]; cause_predictions: [B, N, 2]
        ecp_prob = F.softmax(ecp_predictions, dim=-1)[..., 1]                    # [B, N, N]  P(pair)
        emo_prob = F.softmax(emo_predictions, dim=-1)                            # [B, N, C]
        emo_nonneutral = 1.0 - emo_prob[..., 0]                                  # [B, N]     P(non-neutral)
        cause_prob = F.softmax(cause_predictions, dim=-1)[..., 1]                # [B, N]     P(cause)

        if str(self.config.get('use_method', 'yes')) == 'yes':
            # Method pathway: ecp_prob already is the fused pair probability p_ij (Section 3.8).
            joint = ecp_prob * masks.float()
        else:
            # Legacy joint = P(pair) * P(emo_w ≠ neutral) * P(cause_z)
            joint = ecp_prob * emo_nonneutral.unsqueeze(-1) * cause_prob.unsqueeze(-2)  # [B, N, N]
            joint = joint * masks.float()
        joint_np = joint.cpu().numpy()

        emo_pred = emo_predictions.argmax(-1).cpu().numpy()
        cause_pred = cause_predictions.argmax(-1).cpu().numpy()

        for i in range(len(emo_pred)):
            doc_id = data['doc_ids'][i]
            utt_nums = data['utterance_nums'][i]

            # 情感 / 原因预测仍然使用 argmax (不影响其 P/R/F1)
            emo_pred_ = emo_pred[i, :utt_nums].tolist()
            emo_gold_ = data['labels'][i, :utt_nums].tolist()
            self.preds['emo'] += emo_pred_
            self.golds['emo'] += emo_gold_

            cause_pred_ = cause_pred[i, :utt_nums].tolist()
            cause_gold_ = data['cause_labels'][i, :utt_nums].tolist()
            self.preds['cause'] += cause_pred_
            self.golds['cause'] += cause_gold_

            # 收集所有候选 pair (w >= z, mask=1) 及其 joint score
            jm = joint_np[i, :utt_nums, :utt_nums]
            ws, zs = np.where(jm > 0)
            for w, z in zip(ws, zs):
                if w >= z:
                    self.cand_pairs.append((doc_id, int(w), int(z), float(jm[w, z])))

            pair_num = data['pair_nums'][i]
            self.golds['ecp'] += [(doc_id, *w) for w in data['pairs'][i][:pair_num].tolist()]

    def report_score(self, fixed_threshold=None):
        # === Threshold sweep on collected joint scores ===
        # If fixed_threshold is given (test phase), use it directly without search.
        gold_set = set(self.golds['ecp'])
        cand = self.cand_pairs

        if fixed_threshold is not None:
            best_th = float(fixed_threshold)
            best_pred_set = set((d, w, z) for (d, w, z, s) in cand if s >= best_th)
            tp = len(best_pred_set & gold_set)
            fp = len(best_pred_set - gold_set)
            fn = len(gold_set - best_pred_set)
            best_p = tp / (tp + fp) if tp + fp > 0 else 0
            best_r = tp / (tp + fn) if tp + fn > 0 else 0
            best_f = 2 * best_p * best_r / (best_p + best_r) if best_p + best_r > 0 else 0
        else:
            thresholds = np.arange(0.005, 0.5, 0.005)
            best_f, best_p, best_r, best_th = 0.0, 0.0, 0.0, 0.05
            best_pred_set = set()
            for th in thresholds:
                pred_set = set((d, w, z) for (d, w, z, s) in cand if s >= th)
                tp = len(pred_set & gold_set)
                fp = len(pred_set - gold_set)
                fn = len(gold_set - pred_set)
                p = tp / (tp + fp) if tp + fp > 0 else 0
                r = tp / (tp + fn) if tp + fn > 0 else 0
                f = 2 * p * r / (p + r) if p + r > 0 else 0
                if f > best_f:
                    best_f, best_p, best_r, best_th = f, p, r, th
                    best_pred_set = pred_set

        p, r, f = best_p, best_r, best_f
        self.preds['ecp'] = list(best_pred_set)
        self.best_threshold = best_th
        tp = len(best_pred_set & gold_set)
        fp = len(best_pred_set - gold_set)
        fn = len(gold_set - best_pred_set)

        # 计算情感和原因分数
        gold_emo = [0 if w == 0 else 1 for w in self.golds['emo']]
        pred_emo = [0 if w == 0 else 1 for w in self.preds['emo']]
        emo = precision_recall_fscore_support(gold_emo, pred_emo, average='binary')
        cause = precision_recall_fscore_support(self.golds['cause'], self.preds['cause'], average='binary')
        
        # 生成结果字符串
        res = (f"Pair Pre. {p*100:.4f}\t Rec. {r*100:.4f}\tF1 {f*100:.4f}\t (th={best_th:.3f})\n"
               f"TP {tp}\tPred. {tp+fp}\tGold. {tp+fn}\n"
               f"Emo: Pre. {emo[0]*100:.4f}\t Rec. {emo[1]*100:.4f}\tF1 {emo[2]*100:.4f}\n"
               f"Cause: Pre. {cause[0]*100:.4f}\t Rec. {cause[1]*100:.4f}\tF1 {cause[2]*100:.4f}\n")

        return f, (res, {'p': p, 'r': r, 'default': f, 'emo': emo[2], 'cause': cause[2]})