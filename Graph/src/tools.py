#!/use/bin/env python


import os
import random

import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class AttrDict(dict):
    """Minimal dict with attribute access (config.foo == config['foo']).

    Replaces the external ``attrdict`` package, which is unmaintained and breaks
    on Python >= 3.10 (it imports ABCs removed from ``collections``). Keeping a
    tiny local shim removes a fragile dependency and keeps the project runnable
    on modern Python. ``.get`` / ``.items`` etc. are inherited from ``dict``.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def update_config(config):

    dirs = ['preprocessed_dir', 'target_dir', 'dataset_dir']
    for dirname in dirs:
        if dirname in config:
            config[dirname] = os.path.join(config.data_dir, config[dirname])
    config['emb_file'] = os.path.join(config.preprocessed_dir, config['emb_file'])
    if not os.path.exists(config.preprocessed_dir):
        os.makedirs(config.preprocessed_dir)
    if not os.path.exists(config.target_dir):
        os.makedirs(config.target_dir)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_params_bert(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    bert_params = set(model.bert.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': float(config.bert_lr), 'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'lr': float(config.bert_lr), 'weight_decay': 0.0},
        {'params': other_params, 'lr': float(config.learning_rate), 'weight_decay': float(config.weight_decay)},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.epoch_size * fold_data.__len__())

    config.optimizer = optimizer
    config.scheduler = scheduler

    return config