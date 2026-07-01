#!/usr/bin/env python3
"""
第十一章：Agentic RL完整教学示例

本文件整合了第十一章中介绍的Agentic RL训练流程的所有实用案例:

🎯 核心内容
- SFT (Supervised Fine-Tuning) 训练
- GRPO (Group Relative Policy Optimization) 训练
- 数据集加载和处理
- 奖励函数设计和使用
- LoRA参数高效微调
- 模型评估和分析
- 完整训练流程实战

📚 学习目标:
✅ 理解LLM训练的完整流程(预训练→SFT→RL)
✅ 掌握SFT和GRPO的训练方法
✅ 学会设计和使用奖励函数
✅ 了解LoRA的配置和优化
✅ 掌握模型评估和错误分析方法
✅ 能够构建完整的Agentic RL训练流程

🚀 运行方式:
python examples/chapter11_RL.py

📦 依赖安装:
pip install hello-agents[rl]
# 或者
pip install transformers datasets trl peft accelerate

👨‍💻 作者: HelloAgents 教学团队
📅 更新: 2025年1月
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hello_agents.tools.builtin.rl_training_tool import RLTrainingTool


# ============================================================================
# 示例1: 快速入门 - 最简单的训练流程
# ============================================================================

def example_01_quick_start():
    """
    示例1: 快速入门
    
    最简单的SFT→GRPO→评估流程
    适合初学者快速体验完整训练流程
    
    配置:
    - 模型: Qwen/Qwen3-0.6B (小模型,快速训练)
    - 样本数: 10个 (快速测试)
    - 训练轮数: 1轮
    - 预计时间: 2-3分钟
    """
    print("="*80)
    print("示例1: 快速入门 - 最简单的训练流程")
    print("="*80)
    
    tool = RLTrainingTool()
    
    # 步骤1: SFT训练
    print("\n步骤1: SFT训练(学习基础推理格式)")
    print("-"*80)
    
    sft_result = tool.run({
        "action": "train",
        "algorithm": "sft",
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": "./output/quick_start/sft",
        "max_samples": 10,
        "num_epochs": 1,
        "batch_size": 2,
        "use_lora": True,
    })
    
    print("✅ SFT训练完成!")
    print(json.dumps(json.loads(sft_result), indent=2, ensure_ascii=False))
    
    # 步骤2: GRPO训练
    print("\n步骤2: GRPO训练(强化学习优化)")
    print("-"*80)
    
    grpo_result = tool.run({
        "action": "train",
        "algorithm": "grpo",
        "model_name": "./output/quick_start/sft",
        "output_dir": "./output/quick_start/grpo",
        "max_samples": 5,
        "num_epochs": 1,
        "batch_size": 1,
        "use_lora": True,
    })
    
    print("✅ GRPO训练完成!")
    print(json.dumps(json.loads(grpo_result), indent=2, ensure_ascii=False))
    
    # 步骤3: 评估模型
    print("\n步骤3: 评估训练后的模型")
    print("-"*80)
    
    eval_result = tool.run({
        "action": "evaluate",
        "model_path": "./output/quick_start/grpo",
        "max_samples": 10,
        "use_lora": True,
    })
    
    eval_data = json.loads(eval_result)
    print("✅ 评估完成!")
    print(f"  准确率: {eval_data['accuracy']:.2%}")
    print(f"  平均奖励: {eval_data['average_reward']:.3f}")
    print(f"  测试样本数: {eval_data['num_samples']}")
    
    print("\n" + "="*80)
    print("🎉 快速入门完成!")
    print("="*80)


# ============================================================================
# 示例2: 数据集加载和探索
# ============================================================================

def example_02_dataset_loading():
    """
    示例2: 数据集加载和探索
    
    演示如何加载和查看GSM8K数据集
    了解SFT和RL两种数据格式的区别
    """
    print("="*80)
    print("示例2: 数据集加载和探索")
    print("="*80)
    
    tool = RLTrainingTool()
    
    # 加载SFT格式数据集
    print("\n1. 加载SFT格式数据集")
    print("-"*80)
    
    sft_data = tool.run({
        "action": "load_dataset",
        "format_type": "sft",
        "split": "train",
        "max_samples": 3,
    })
    
    print("SFT数据格式:")
    print(json.dumps(json.loads(sft_data), indent=2, ensure_ascii=False))
    
    # 加载RL格式数据集
    print("\n2. 加载RL格式数据集")
    print("-"*80)
    
    rl_data = tool.run({
        "action": "load_dataset",
        "format_type": "rl",
        "split": "train",
        "max_samples": 3,
    })
    
    print("RL数据格式:")
    print(json.dumps(json.loads(rl_data), indent=2, ensure_ascii=False))
    
    print("\n" + "="*80)
    print("数据集加载完成!")
    print("="*80)


# ============================================================================
# 示例3: 奖励函数设计
# ============================================================================

def example_03_reward_functions():
    """
    示例3: 奖励函数设计
    
    演示不同类型的奖励函数:
    - 准确率奖励 (accuracy)
    - 长度惩罚奖励 (length_penalty)
    - 步骤奖励 (step)
    """
    print("="*80)
    print("示例3: 奖励函数设计")
    print("="*80)
    
    tool = RLTrainingTool()
    
    # 1. 准确率奖励
    print("\n1. 准确率奖励")
    print("-"*80)
    
    accuracy_reward = tool.run({
        "action": "create_reward",
        "reward_type": "accuracy",
    })
    
    print("准确率奖励函数:")
    print(json.dumps(json.loads(accuracy_reward), indent=2, ensure_ascii=False))
    
    # 2. 长度惩罚奖励
    print("\n2. 长度惩罚奖励")
    print("-"*80)
    
    length_reward = tool.run({
        "action": "create_reward",
        "reward_type": "length_penalty",
        "penalty_weight": 0.01,
    })
    
    print("长度惩罚奖励函数:")
    print(json.dumps(json.loads(length_reward), indent=2, ensure_ascii=False))
    
    # 3. 步骤奖励
    print("\n3. 步骤奖励")
    print("-"*80)
    
    step_reward = tool.run({
        "action": "create_reward",
        "reward_type": "step",
        "step_bonus": 0.1,
    })
    
    print("步骤奖励函数:")
    print(json.dumps(json.loads(step_reward), indent=2, ensure_ascii=False))
    
    print("\n" + "="*80)
    print("奖励函数创建完成!")
    print("="*80)


# ============================================================================
# 示例4: LoRA配置优化
# ============================================================================

def example_04_lora_configuration():
    """
    示例4: LoRA配置优化
    
    演示不同LoRA配置的效果:
    - 快速实验配置 (r=8)
    - 标准配置 (r=16)
    - 高性能配置 (r=32)
    """
    print("="*80)
    print("示例4: LoRA配置优化")
    print("="*80)
    
    tool = RLTrainingTool()
    
    configs = {
        "快速实验": {"lora_r": 8, "lora_alpha": 16, "batch_size": 8},
        "标准配置": {"lora_r": 16, "lora_alpha": 32, "batch_size": 4},
        "高性能": {"lora_r": 32, "lora_alpha": 64, "batch_size": 2},
    }
    
    print("\nLoRA配置对比:")
    print("-"*80)
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  lora_r: {config['lora_r']}")
        print(f"  lora_alpha: {config['lora_alpha']}")
        print(f"  batch_size: {config['batch_size']}")
    
    # 使用标准配置进行训练
    print("\n使用标准配置进行训练:")
    print("-"*80)
    
    result = tool.run({
        "action": "train",
        "algorithm": "sft",
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": "./output/lora_standard",
        "max_samples": 10,
        "num_epochs": 1,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 4,
    })
    
    print("✅ 训练完成!")
    print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))
    
    print("\n" + "="*80)
    print("LoRA配置优化完成!")
    print("="*80)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 运行所有示例"""
    print("\n" + "🎓 "*20)
    print("第十一章: Agentic RL 完整教学示例")
    print("🎓 "*20 + "\n")
    
    examples = [
        ("示例1: 快速入门", example_01_quick_start),
        ("示例2: 数据集加载", example_02_dataset_loading),
        ("示例3: 奖励函数设计", example_03_reward_functions),
        ("示例4: LoRA配置优化", example_04_lora_configuration),
    ]
    
    print("可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n选择要运行的示例 (输入数字,或按Enter运行所有示例):")
    choice = input("> ").strip()
    
    if choice == "":
        # 运行所有示例
        for name, func in examples:
            print(f"\n\n{'='*80}")
            print(f"运行: {name}")
            print('='*80)
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        # 运行选定的示例
        name, func = examples[int(choice) - 1]
        print(f"\n\n{'='*80}")
        print(f"运行: {name}")
        print('='*80)
        func()
    else:
        print("❌ 无效的选择!")
        return
    
    print("\n\n" + "🎉 "*20)
    print("所有示例运行完成!")
    print("🎉 "*20 + "\n")


if __name__ == "__main__":
    main()

