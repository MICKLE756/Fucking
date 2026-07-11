"""SWE-bench Lite 评测入口。

流程：
1. 从 HuggingFace 加载 princeton-nlp/SWE-bench_Lite 数据集；
2. 对每个 instance：克隆对应仓库并 checkout 到 base_commit，运行 Agent 产出 patch；
3. 输出官方评测格式的 predictions.jsonl；
4. 用官方 harness 计算 resolve rate：
   python -m swebench.harness.run_evaluation \
       --dataset_name princeton-nlp/SWE-bench_Lite \
       --predictions_path predictions.jsonl \
       --max_workers 4 --run_id my_run

用法：
    python -m sweagent0.eval.swebench --config config.yaml --limit 10 --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from ..agent import Agent
from ..config import RunConfig

logger = logging.getLogger("sweagent0.eval")

MODEL_NAME_FIELD = "sweagent0"


def load_instances(split: str = "test", limit: int | None = None) -> list[dict]:
    from datasets import load_dataset  # 重依赖，按需导入

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    instances = list(ds)
    return instances[:limit] if limit else instances


def prepare_repo(instance: dict, workspace: Path) -> Path:
    """克隆 instance 对应仓库并 checkout 到 base_commit。"""
    repo = instance["repo"]  # e.g. "django/django"
    repo_dir = workspace / repo.replace("/", "__")
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(repo_dir)],
            check=True,
            capture_output=True,
        )
    subprocess.run(["git", "checkout", "-f", instance["base_commit"]], cwd=repo_dir, check=True,
                   capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, check=True, capture_output=True)
    return repo_dir


def run_instance(instance: dict, config: RunConfig, workspace: Path) -> dict:
    repo_dir = prepare_repo(instance, workspace)
    config.workdir = str(repo_dir)
    agent = Agent(config)
    result = agent.run(instance["problem_statement"])
    traj_dir = workspace / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    result.trajectory.save(traj_dir / f"{instance['instance_id']}.json")
    logger.info(
        "%s: status=%s steps=%d patch_bytes=%d",
        instance["instance_id"], result.status, result.steps_used, len(result.patch),
    )
    return {
        "instance_id": instance["instance_id"],
        "model_name_or_path": MODEL_NAME_FIELD,
        "model_patch": result.patch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在 SWE-bench Lite 上运行 sweagent0")
    parser.add_argument("--config", default=None, help="RunConfig YAML 路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个 instance")
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--workspace", default=None, help="仓库克隆工作区（默认临时目录）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = RunConfig.from_yaml(args.config) if args.config else RunConfig()
    workspace = Path(args.workspace) if args.workspace else Path(tempfile.mkdtemp(prefix="sweb-"))
    workspace.mkdir(parents=True, exist_ok=True)

    instances = load_instances(limit=args.limit)
    with open(args.output, "a", encoding="utf-8") as f:
        for instance in instances:
            try:
                pred = run_instance(instance, config, workspace)
            except Exception:
                logger.exception("instance %s 运行失败", instance["instance_id"])
                pred = {
                    "instance_id": instance["instance_id"],
                    "model_name_or_path": MODEL_NAME_FIELD,
                    "model_patch": "",
                }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f.flush()
    print(f"预测结果已写入 {args.output}，用官方 harness 计算 resolve rate（见模块 docstring）")


if __name__ == "__main__":
    main()
