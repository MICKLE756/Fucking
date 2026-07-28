#!/usr/bin/env python

"""
Name: loader.py  (ConvECPE / IEMOCAP adaptation)

Loads the ConvECPE dataset (Dataset/IEMOCAP_emotion_cause_features.pkl, from
https://github.com/Maxwe11y/JointEC) and produces batches in the same tensor
layout as the ECF pipeline in ../src, so the method-chapter model runs
unchanged on top of it.

Pickle layout (list of 12 entries):
  0 videoIDs        {vid: [utt_id, ...]}
  1 videoSpeakers   {vid: ['M'/'F', ...]}
  2 videoLabels     {vid: [emotion 0-5, ...]}   (0 hap, 1 sad, 2 neu, 3 ang, 4 exc, 5 fru)
  3-5 causeLabels   {vid: [cause utt index (1-based, 0 = none), ...]} x3
  6 videoText       {vid: [100-d]}   (unused: raw sentences are re-encoded by BERT)
  7 videoAudio      {vid: [100-d]}
  8 videoVisual     {vid: [512-d]}
  9 videoSentence   {vid: [str, ...]}
  10 trainVid (set) 11 testVid (set)
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import logging
import pickle as pkl
from typing import Dict
from dataclasses import dataclass
import numpy as np

# IEMOCAP label index -> emotion name; the label dict below puts neutral at
# index 0, matching the convention the trainer and model assume.
IDX2EMO = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'angry', 4: 'excited', 5: 'frustrated'}
LABEL_DICT = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'excited': 4, 'frustrated': 5}
SPEAKER_DICT = {'M': 0, 'F': 1}


def build_mask(utterance_nums, speakers):
    max_utterance = max(utterance_nums)

    gmask = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(utterance_nums)):
        gmask[i, :utterance_nums[i], :utterance_nums[i]] = 1
    gmask = gmask.repeat(1, 4, 4)

    smask = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(speakers)):
        speaker = speakers[i]
        m = np.array([[1 if i == j else 0 for i in speaker] for j in speaker])
        smask[i, :utterance_nums[i], :utterance_nums[i]] = torch.tensor(m)
    smask = smask.repeat(1, 4, 4)

    rmask = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(utterance_nums)):
        utterance_num = utterance_nums[i]
        eye = np.eye(utterance_num) + np.eye(utterance_num, k=1) + np.eye(utterance_num, k=-1)
        rmask[i, :utterance_nums[i], :utterance_nums[i]] = torch.tensor(eye, dtype=torch.long)
    rmask = rmask.repeat(1, 4, 4)
    return gmask, smask, rmask


class SupervisedDataset(Dataset):
    """Dataset over pre-structured ConvECPE dialogues."""

    def __init__(self, data, mode):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.data = data[mode]
        self.label_dict = data['label_dict']
        self.speaker_dict = data['speaker_dict']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        keys = list(self.data[i])
        values = [self.data[i][k] for k in keys]
        # values: [doc_id, emotion_cause_pairs, lines, speakers, emotions, audio, video]
        values[4] = [self.label_dict[w] for w in values[4]]
        values[3] = [self.speaker_dict[w] for w in values[3]]
        return (keys, values)


@dataclass
class CollateFN:
    """Collate ConvECPE dialogues into the tensor layout the model expects."""

    tokenizer: transformers.PreTrainedTokenizer
    config: Dict

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        keys, instances = list(zip(*instances))
        doc_ids = [line[0] for line in instances]
        pairs = [[(w - 1, z - 1) for w, z in line[1]] for line in instances]
        utterances = [line[2] for line in instances]
        speakers = [line[3] for line in instances]
        emotions = [line[4] for line in instances]
        audio_list = [line[5] for line in instances]
        video_list = [line[6] for line in instances]
        label_dict = self.config['label_dict']

        if self.config.emo_cat != 'yes':
            emotions = [[0 if w == label_dict['neutral'] else 1 for w in line] for line in emotions]

        utterance_nums = [len(line) for line in utterances]
        gmasks, smasks, rmasks = build_mask(utterance_nums, speakers)
        IGNORE_INDEX = -100
        max_utterance = max(utterance_nums)

        emotions = [w + [IGNORE_INDEX] * (max_utterance - len(w)) for w in emotions]
        max_length = self.config['max_length']
        total_length = self.config['total_length']
        input_tokens, indices = pack(utterances, max_length, total_length, self.tokenizer, self.config)

        max_seq_len = max([len(w) for w in input_tokens])
        input_tokens = [w + [self.config.pad] * (max_seq_len - len(w)) for w in input_tokens]

        input_ids = [self.tokenizer.convert_tokens_to_ids(w) for w in input_tokens]
        attention_mask = [[1] * len(w) + [0] * (max_seq_len - len(w)) for w in input_tokens]

        # padding pairs
        pair_nums = [len(line) for line in pairs]
        max_pair = max(max(pair_nums), 1)
        pairs = [w + [(IGNORE_INDEX, IGNORE_INDEX)] * (max_pair - len(w)) for w in pairs]

        cause_labels = [[0 for _ in range(max_utterance)] for _ in range(len(pairs))]
        for i in range(len(pairs)):
            for w, z in pairs[i]:
                if z != IGNORE_INDEX:
                    cause_labels[i][z] = 1

        speakers = [w + [0] * (max_utterance - len(w)) for w in speakers]

        def pad_features(feature_list):
            feats = []
            for w in feature_list:
                w = np.asarray(w, dtype=np.float32)
                feats.append(np.concatenate([w, np.zeros((max_utterance - w.shape[0], w.shape[1]), dtype=np.float32)], axis=0))
            return np.clip(np.stack(feats), -1, 1)

        audio_features = pad_features(audio_list)
        video_features = pad_features(video_list)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long).to(self.config['device']),
            'input_masks': torch.tensor(attention_mask, dtype=torch.long).to(self.config['device']),
            'indices': indices,  # [(global_id, start, end), ...]
            'utterance_nums': torch.tensor(utterance_nums, dtype=torch.long).to(self.config['device']),
            'pairs': torch.tensor(pairs, dtype=torch.long).to(self.config['device']),
            'pair_nums': torch.tensor(pair_nums, dtype=torch.long).to(self.config['device']),
            'labels': torch.tensor(emotions, dtype=torch.long).to(self.config['device']),
            'cause_labels': torch.tensor(cause_labels, dtype=torch.long).to(self.config['device']),
            'doc_ids': doc_ids,
            'speaker_ids': torch.tensor(speakers, dtype=torch.long).to(self.config['device']),
            'video_features': torch.tensor(video_features, dtype=torch.float).to(self.config['device']),
            'audio_features': torch.tensor(audio_features, dtype=torch.float).to(self.config['device']),
            'gmasks': gmasks.to(self.config['device']),
            'smasks': smasks.to(self.config['device']),
            'rmasks': rmasks.to(self.config['device']),
        }


def pack(dialogues, max_len, total_len, tokenizer, config):
    res = []
    indices = []
    for i in range(len(dialogues)):
        cur_res = [config.cls]
        cur_indices = []
        for line in dialogues[i]:
            tokens = tokenizer.tokenize(line)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            if len(cur_res) + len(tokens) > total_len:
                res.append(cur_res)
                cur_res = [config.cls]
            global_id = len(res)
            start = len(cur_res)
            cur_res += tokens + [config.sep]
            end = len(cur_res) - 1
            cur_indices.append((global_id, start, end))
        res.append(cur_res)
        indices.append(cur_indices)
    return res, indices


def read_convecpe(path):
    """Read the JointEC pickle into per-split structured dialogue lists.

    Emotion-cause pairs are 1-based (emotion_idx, cause_idx); an utterance can
    have up to three annotated causes (the three causeLabels fields). doc_id is
    an integer index over the dialogue ids (sorted for determinism).
    """
    with open(path, 'rb') as f:
        raw = pkl.load(f)
    (video_ids, video_speakers, video_labels, cause1, cause2, cause3,
     _video_text, video_audio, video_visual, video_sentence,
     train_vids, test_vids) = raw

    vid_index = {vid: i for i, vid in enumerate(sorted(video_ids))}

    def build(vids):
        dialogues = []
        for vid in sorted(vids):
            n = len(video_ids[vid])
            pairs = []
            for i in range(n):
                for cfield in (cause1, cause2, cause3):
                    c = int(cfield[vid][i])
                    if c > 0:
                        pairs.append((i + 1, c))
            dialogues.append({
                'doc_id': vid_index[vid],
                'emotion_cause_pairs': pairs,
                'lines': list(video_sentence[vid]),
                'speakers': list(video_speakers[vid]),
                'emotions': [IDX2EMO[int(w)] for w in video_labels[vid]],
                'audio': np.asarray(video_audio[vid], dtype=np.float32),
                'video': np.asarray(video_visual[vid], dtype=np.float32),
            })
        return dialogues

    # carve a fixed validation split out of train (threshold selection protocol)
    train_list = sorted(train_vids)
    rng = random.Random(13)
    rng.shuffle(train_list)
    n_valid = max(1, int(round(len(train_list) * 0.1)))
    valid_vids, train_vids_ = train_list[:n_valid], train_list[n_valid:]

    return {
        'train': build(train_vids_),
        'valid': build(valid_vids),
        'test': build(sorted(test_vids)),
        'label_dict': dict(LABEL_DICT),
        'speaker_dict': dict(SPEAKER_DICT),
    }


def make_supervised_data_module(config, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for ConvECPE supervised training."""
    cache = os.path.join(config.preprocessed_dir, 'convecpe.pkl')
    if not os.path.exists(cache):
        data = read_convecpe(config.dataset_path)
        with open(cache, 'wb') as f:
            pkl.dump(data, f)
    else:
        with open(cache, 'rb') as f:
            data = pkl.load(f)
    config['label_dict'] = data['label_dict']
    config['speaker_dict'] = data['speaker_dict']
    train_dataset = SupervisedDataset(data, 'train')
    valid_dataset = SupervisedDataset(data, 'valid')
    test_dataset = SupervisedDataset(data, 'test')
    data_collator = CollateFN(tokenizer=tokenizer, config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=data_collator)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=data_collator)

    return train_loader, valid_loader, test_loader, config
