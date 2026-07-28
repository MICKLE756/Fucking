#!/usr/bin/env python

"""ConvECPE (IEMOCAP) entry point.

Run from the Graph/ directory:
    python main_conv.py
    python main_conv.py --set use_necessity=no --seed 1
"""

import argparse
import yaml
import random
import string
import torch
import warnings

from ConvECPE_src.tools import update_config, set_seed, load_params_bert, AttrDict


def _coerce(raw):
    """Type-coerce a --set value while keeping 'yes'/'no' flag strings intact."""
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    low = raw.strip().lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null'):
        return None
    return raw


from ConvECPE_src.trainer import MyTrainer
from ConvECPE_src.loader import make_supervised_data_module
import transformers
from ConvECPE_src.model import TextClassification

warnings.filterwarnings('ignore')


class Template:
    def __init__(self, args):
        config_path = getattr(args, 'config', 'ConvECPE_src/config.yaml')
        config = AttrDict(yaml.load(
            open(config_path, 'r', encoding='utf-8'),
            Loader=yaml.FullLoader
        ))

        for k, v in vars(args).items():
            if k in ('set', 'config') or v is None:
                continue
            setattr(config, k, v)
        for item in (getattr(args, 'set', None) or []):
            if '=' not in item:
                raise ValueError(f"--set expects key=value, got: {item}")
            key, raw = item.split('=', 1)
            config[key.strip()] = _coerce(raw)
        config = update_config(config)

        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        config.save_name = f"{config.model_name}_{random_str}_{config.seed}_{{}}.pt"

        set_seed(config.seed)
        config.device = torch.device(f'cuda:{config.cuda_index}' if torch.cuda.is_available() else 'cpu')

        self.config = config

    def forward(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.bert_path,
            padding_side="right",
            use_fast=False
        )

        self.train_loader, self.valid_loader, self.test_loader, self.config = \
            make_supervised_data_module(self.config, tokenizer)

        if self.config.model_name == 'bert':
            self.model = TextClassification(self.config, tokenizer).to(self.config.device)

        self.config = load_params_bert(self.config, self.model, self.train_loader)

        trainer = MyTrainer(self.model, self.config, self.train_loader, self.valid_loader, self.test_loader)
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert', help='model type')
    parser.add_argument('-cd', '--cuda_index', type=int, default=0, help='cuda device index')
    parser.add_argument('--config', type=str, default='ConvECPE_src/config.yaml',
                        help='config file path')
    parser.add_argument('--seed', type=int, default=None,
                        help='override the random seed from the config (multi-seed runs)')
    parser.add_argument('--set', action='append', metavar='KEY=VALUE', default=None,
                        help='override any config field, e.g. --set use_necessity=no')

    template = Template(parser.parse_args())
    template.forward()
