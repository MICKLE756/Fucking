#!/usr/bin/env python

import argparse
import yaml
import random
import string
import torch
import warnings

from src.tools import update_config, set_seed, load_params_bert, AttrDict


def _coerce(raw):
    """Type-coerce a --set value while keeping 'yes'/'no' flag strings intact.

    NOTE: we deliberately do NOT use yaml here, because YAML 1.1 parses 'no'/'yes'
    as booleans, which would silently break the 'yes'/'no' ablation switches."""
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


from src.trainer import MyTrainer 
from src.loader import make_supervised_data_module
import transformers
from src.model import TextClassification

warnings.filterwarnings('ignore')

class Template:
    def __init__(self, args):
        # 加载配置文件（默认 src/config.yaml；可用 --config 指定精简档 src/config_lite.yaml）
        config_path = getattr(args, 'config', 'src/config.yaml')
        config = AttrDict(yaml.load(
            open(config_path, 'r', encoding='utf-8'), 
            Loader=yaml.FullLoader
        ))
        
        # 更新配置（仅覆盖显式提供的命令行参数，避免用默认 None 冲掉 yaml）
        for k, v in vars(args).items():
            if k in ('set', 'config') or v is None:
                continue
            setattr(config, k, v)
        # --set key=value 行内覆盖任意 config 字段（用于消融/多种子扫描）
        for item in (getattr(args, 'set', None) or []):
            if '=' not in item:
                raise ValueError(f"--set expects key=value, got: {item}")
            key, raw = item.split('=', 1)
            # 类型推断：3->int, 0.1->float, true/false->bool, 其余(含 yes/no)保持字符串
            config[key.strip()] = _coerce(raw)
        config = update_config(config)
        
        # 设置模型保存名称
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        config.save_name = f"{config.model_name}_{random_str}_{config.seed}_{{}}.pt"
        
        # 设置随机种子和设备
        set_seed(config.seed)
        config.device = torch.device(f'cuda:{config.cuda_index}' if torch.cuda.is_available() else 'cpu')
        
        self.config = config

    def forward(self):
        # 初始化tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.bert_path, 
            padding_side="right",
            use_fast=False
        )

        # 准备数据
        self.train_loader, self.valid_loader, self.test_loader, self.config = \
            make_supervised_data_module(self.config, tokenizer)

        # 初始化模型
        if self.config.model_name == 'bert':
            self.model = TextClassification(self.config, tokenizer).to(self.config.device)

        # 加载优化器等参数
        self.config = load_params_bert(self.config, self.model, self.train_loader) 

        # 训练模型
        trainer = MyTrainer(self.model, self.config, self.train_loader, self.valid_loader, self.test_loader)
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert', help='model type')
    parser.add_argument('-cd', '--cuda_index', type=int, default=0, help='cuda device index')
    parser.add_argument('--config', type=str, default='src/config.yaml',
                        help='config file path; use src/config_lite.yaml for the trimmed preset')
    parser.add_argument('--seed', type=int, default=None,
                        help='override the random seed from the config (multi-seed runs)')
    parser.add_argument('--set', action='append', metavar='KEY=VALUE', default=None,
                        help='override any config field, e.g. --set use_necessity=no --set epoch_size=25')
    
    template = Template(parser.parse_args())
    template.forward()
