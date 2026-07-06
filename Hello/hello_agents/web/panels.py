"""终极 dashboard 的数据构建器。

只在 web 层使用；不修改框架内部模块。
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tiktoken

from hello_agents import __version__
from hello_agents.protocols.mcp.utils import create_context, parse_context
from hello_agents.protocols.a2a import A2AServer, AgentNetwork, AgentRegistry
from hello_agents.protocols.anp import ANPDiscovery, ANPNetwork, ServiceInfo, register_service, discover_service


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {rel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


BFCL = _load_module("web_bfcl_metrics", "hello_agents/evaluation/benchmarks/bfcl/metrics.py")
GAIA = _load_module("web_gaia_metrics", "hello_agents/evaluation/benchmarks/gaia/metrics.py")
RL_UTILS = _load_module("web_rl_utils", "hello_agents/rl/utils.py")
RL_REWARDS = _load_module("web_rl_rewards", "hello_agents/rl/rewards.py")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (set,)):
        return sorted(list(value))
    return value


def sanitize(value: Any) -> Any:
    return _jsonable(value)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _now(offset: int = 0) -> str:
    return _iso(datetime.now() - timedelta(hours=offset))


def _encoding():
    try:
        return tiktoken.encoding_for_model("gpt-4")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    try:
        return len(_encoding().encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _memory_items() -> Dict[str, Any]:
    working = [
        {
            "id": "wm-001",
            "content": "用户偏好：以后回复尽量先给结论，再给 3 条要点，语气保持简洁专业。",
            "memory_type": "working",
            "user_id": "user_007",
            "importance": 0.92,
            "timestamp": _now(1),
            "metadata": {"session_id": "sess-042", "source": "dialog", "tags": ["偏好", "风格"]},
        },
        {
            "id": "wm-002",
            "content": "当前任务：准备一个终极 dashboard，覆盖 memory/context/protocols/rl/eval 五大模块。",
            "memory_type": "working",
            "user_id": "user_007",
            "importance": 0.88,
            "timestamp": _now(2),
            "metadata": {"session_id": "sess-042", "source": "task", "tags": ["dashboard", "planning"]},
        },
        {
            "id": "wm-003",
            "content": "工具调用结果：calculate(15*23+45)=390，后续可复用此结果回答数学题。",
            "memory_type": "working",
            "user_id": "user_007",
            "importance": 0.76,
            "timestamp": _now(3),
            "metadata": {"tool": "calculate", "session_id": "sess-042"},
        },
    ]
    episodic = [
        {
            "id": "ep-001",
            "content": "昨天用户提到：项目演示要优先显示“真实/演示”标签，并保留 SSE ReAct 流程。",
            "memory_type": "episodic",
            "user_id": "user_007",
            "importance": 0.84,
            "timestamp": _iso(datetime.now() - timedelta(days=1, hours=3)),
            "metadata": {"session_id": "sess-039", "outcome": "dashboard requirement captured", "tags": ["偏好", "展示"]},
        },
        {
            "id": "ep-002",
            "content": "上次会话中，用户确认了左侧侧边栏导航和中文 UI 文案风格。",
            "memory_type": "episodic",
            "user_id": "user_007",
            "importance": 0.78,
            "timestamp": _iso(datetime.now() - timedelta(days=2, hours=5)),
            "metadata": {"session_id": "sess-038", "outcome": "layout accepted", "tags": ["UI", "sidebar"]},
        },
        {
            "id": "ep-003",
            "content": "本周一次调试中，A2A Agent 网络成功发现了 3 个节点并完成一跳路由。",
            "memory_type": "episodic",
            "user_id": "user_007",
            "importance": 0.71,
            "timestamp": _iso(datetime.now() - timedelta(days=4, hours=1)),
            "metadata": {"session_id": "sess-035", "outcome": "network route validated", "tags": ["A2A", "ANP"]},
        },
    ]
    semantic = [
        {
            "id": "sm-001",
            "content": "HelloAgents 是一个多智能体框架，memory、context、protocols、rl、evaluation 均是顶层能力域。",
            "memory_type": "semantic",
            "user_id": "user_007",
            "importance": 0.66,
            "timestamp": _now(6),
            "metadata": {
                "triples": [
                    {"subject": "HelloAgents", "predicate": "包含", "object": "memory"},
                    {"subject": "HelloAgents", "predicate": "包含", "object": "context"},
                    {"subject": "HelloAgents", "predicate": "包含", "object": "protocols"},
                ],
                "entity_graph": ["HelloAgents", "memory", "context", "protocols"],
            },
        },
        {
            "id": "sm-002",
            "content": "MCP 的上下文对象由 messages、tools、resources、metadata 四段组成。",
            "memory_type": "semantic",
            "user_id": "user_007",
            "importance": 0.73,
            "timestamp": _now(7),
            "metadata": {
                "triples": [
                    {"subject": "MCP context", "predicate": "has_part", "object": "messages"},
                    {"subject": "MCP context", "predicate": "has_part", "object": "tools"},
                    {"subject": "MCP context", "predicate": "has_part", "object": "resources"},
                    {"subject": "MCP context", "predicate": "has_part", "object": "metadata"},
                ]
            },
        },
        {
            "id": "sm-003",
            "content": "ANP 是服务发现 + 网络拓扑 + 消息路由的概念性协议实现。",
            "memory_type": "semantic",
            "user_id": "user_007",
            "importance": 0.69,
            "timestamp": _now(8),
            "metadata": {
                "triples": [
                    {"subject": "ANP", "predicate": "supports", "object": "service discovery"},
                    {"subject": "ANP", "predicate": "supports", "object": "routing"},
                ]
            },
        },
    ]
    perceptual = [
        {
            "id": "pm-001",
            "content": "图像记忆：一张深色主题仪表盘草图，左侧有导航栏，右侧是卡片式内容区域。",
            "memory_type": "perceptual",
            "user_id": "user_007",
            "importance": 0.62,
            "timestamp": _now(4),
            "metadata": {"modality": "image", "raw_data": "dashboard_sketch.png", "tags": ["UI", "image"]},
        },
        {
            "id": "pm-002",
            "content": "音频记忆：会议录音摘要，用户强调“所有子系统都要可视化，且必须区分 live 与 demo”。",
            "memory_type": "perceptual",
            "user_id": "user_007",
            "importance": 0.74,
            "timestamp": _now(5),
            "metadata": {"modality": "audio", "raw_data": "meeting_clip.wav", "tags": ["requirements", "audio"]},
        },
        {
            "id": "pm-003",
            "content": "文本感知：用户曾上传一段错误日志，提示 scikit-learn 不可用导致记忆模块导入失败。",
            "memory_type": "perceptual",
            "user_id": "user_007",
            "importance": 0.58,
            "timestamp": _now(9),
            "metadata": {"modality": "text", "raw_data": "log_excerpt.txt", "tags": ["debug", "error"]},
        },
    ]

    all_items = working + episodic + semantic + perceptual
    avg_importance = round(sum(item["importance"] for item in all_items) / len(all_items), 3)
    stats = {
        "count": len(all_items),
        "total_count": len(all_items),
        "avg_importance": avg_importance,
        "capacity_usage": round(len(all_items) / 100, 3),
        "token_usage": round(sum(count_tokens(item["content"]) for item in all_items) / 2000, 3),
        "memories_by_type": {
            "working": {"count": len(working), "total_count": len(working), "memory_type": "working"},
            "episodic": {"count": len(episodic), "total_count": len(episodic), "memory_type": "episodic"},
            "semantic": {"count": len(semantic), "total_count": len(semantic), "memory_type": "semantic"},
            "perceptual": {"count": len(perceptual), "total_count": len(perceptual), "memory_type": "perceptual"},
        },
    }
    return {
        "source": "demo",
        "working": working,
        "episodic": episodic,
        "semantic": semantic,
        "perceptual": perceptual,
        "stats": stats,
        "consolidation": {
            "from_type": "working",
            "to_type": "episodic",
            "importance_threshold": 0.7,
            "selected_ids": ["wm-001", "wm-002"],
            "result": "2 条高重要性工作记忆可整合到情景记忆层",
        },
        "forgetting": {
            "strategy": "importance_based",
            "threshold": 0.1,
            "would_forget": [],
            "result": "当前样本均高于阈值，暂无遗忘项",
        },
    }


def _context_builder() -> Dict[str, Any]:
    max_tokens = 8000
    reserve_ratio = 0.15
    available_tokens = int(max_tokens * (1 - reserve_ratio))

    packets_source = [
        ("instructions", "请优先给出结论，再列出要点，并保留关键证据。"),
        ("task_state", "当前任务：构建终极 dashboard，覆盖 memory/context/protocols/rl/eval。"),
        ("related_memory", "用户偏好：中文输出、左侧导航、live/demo 标识必须显眼。"),
        ("knowledge_base", "MCP 上下文结构：messages / tools / resources / metadata。"),
        ("history", "最近一次 ReAct 调试中，calculate 工具已成功返回 390。"),
        ("tool_result", "ANP 网络示例：5 个服务，3 跳以内可完成路由。"),
        ("retrieval", "BFCL metrics 可以直接计算 accuracy、score distribution 和 function call stats。"),
    ]
    packets = []
    for idx, (kind, content) in enumerate(packets_source, 1):
        packets.append(
            {
                "content": content,
                "timestamp": _iso(datetime.now() - timedelta(minutes=idx * 7)),
                "token_count": count_tokens(content),
                "relevance_score": round(0.9 - idx * 0.08, 2),
                "metadata": {"type": kind, "packet_id": f"pkt-{idx:02d}"},
            }
        )

    gather_tokens = sum(p["token_count"] for p in packets)
    selected = sorted(packets, key=lambda p: (p["metadata"]["type"] == "instructions", p["relevance_score"]), reverse=True)
    selected = [p for p in selected if p["relevance_score"] >= 0.3]
    selected_tokens = sum(p["token_count"] for p in selected)

    sections = OrderedDict(
        [
            ("[Role & Policies]", [p["content"] for p in packets if p["metadata"]["type"] == "instructions"]),
            ("[Task]", ["用户问题：请设计一个可视化终极 dashboard。"]),
            ("[State]", [p["content"] for p in packets if p["metadata"]["type"] == "task_state"]),
            (
                "[Evidence]",
                [p["content"] for p in packets if p["metadata"]["type"] in {"related_memory", "knowledge_base", "retrieval", "tool_result"}],
            ),
            ("[Context]", [p["content"] for p in packets if p["metadata"]["type"] == "history"]),
            (
                "[Output]",
                [
                    "1. 结论（简洁明确）",
                    "2. 依据（列出支撑证据及来源）",
                    "3. 风险与假设（如有）",
                    "4. 下一步行动建议（如适用）",
                ],
            ),
        ]
    )
    structured = "\n\n".join(f"{title}\n" + "\n".join(lines) for title, lines in sections.items() if lines)
    structured_tokens = count_tokens(structured)

    final_context = structured
    if structured_tokens > available_tokens:
        lines = structured.splitlines()
        kept = []
        used = 0
        for line in lines:
            tok = count_tokens(line)
            if used + tok > available_tokens:
                break
            kept.append(line)
            used += tok
        final_context = "\n".join(kept)

    final_tokens = count_tokens(final_context)
    return {
        "source": "live",
        "config": {
            "max_tokens": max_tokens,
            "reserve_ratio": reserve_ratio,
            "available_tokens": available_tokens,
            "min_relevance": 0.3,
            "enable_mmr": True,
            "enable_compression": True,
        },
        "packets": packets,
        "pipeline": [
            {"stage": "Gather", "packets": len(packets), "tokens_in": gather_tokens, "tokens_out": gather_tokens},
            {"stage": "Select", "packets": len(selected), "tokens_in": gather_tokens, "tokens_out": selected_tokens},
            {"stage": "Structure", "packets": len(selected), "tokens_in": selected_tokens, "tokens_out": structured_tokens},
            {"stage": "Compress", "packets": len(selected), "tokens_in": structured_tokens, "tokens_out": final_tokens},
        ],
        "final_prompt": final_context,
        "budget": {
            "used_tokens": final_tokens,
            "available_tokens": available_tokens,
            "usage_ratio": round(final_tokens / available_tokens, 3) if available_tokens else 0.0,
        },
        "history": {
            "min_retain_rounds": 10,
            "compression_threshold": 0.8,
            "total_rounds": 13,
            "retained_rounds": 10,
            "summary": "前 3 轮历史已压缩为摘要：用户要求中文 UI、侧边栏导航、9 个面板，并强调 live/demo 标记和 ReAct SSE 保留。",
        },
        "truncation": {
            "max_lines": 2000,
            "max_bytes": 51200,
            "direction": "head",
            "before": {"lines": 2438, "bytes": 82144},
            "after": {
                "truncated": True,
                "preview": "tool-output-001: 第 1 行\n...\ntool-output-001: 第 2000 行",
                "full_output_path": "/tmp/tool-output/tool_20250706_000001_demo.json",
            },
        },
    }


def _mcp_builder() -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "你是一个支持多工具协作的助手。"},
        {"role": "user", "content": "请为今天的 dashboard 生成一个简要说明。"},
    ]
    tools = [
        {
            "name": "calculate",
            "description": "执行数学计算",
            "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        },
        {
            "name": "search",
            "description": "搜索网页或文档",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    ]
    resources = [
        {"uri": "file://memory-summary", "name": "Memory Summary", "mimeType": "text/plain"},
        {"uri": "file://protocol-map", "name": "Protocol Map", "mimeType": "application/json"},
    ]
    metadata = {"session_id": "mcp-demo-001", "source": "hello_agents.web"}

    built = create_context(messages=messages, tools=tools, resources=resources, metadata=metadata)
    parsed = parse_context(json.dumps(built, ensure_ascii=False))
    return {
        "source": "live",
        "explanation": [
            {"key": "messages", "description": "对话消息上下文，包含 system/user/assistant 轮次。"},
            {"key": "tools", "description": "可调用工具列表及其参数 schema。"},
            {"key": "resources", "description": "可引用资源，如文件、文档或外部对象。"},
            {"key": "metadata", "description": "会话级附加信息，例如 session_id、来源等。"},
        ],
        "built_context": built,
        "parsed_context": parsed,
        "round_trip_equal": built == parsed,
    }


def _a2a_builder() -> Dict[str, Any]:
    def _skill_reply(prefix: str):
        def _fn(text: str) -> str:
            return f"{prefix} 已收到：{text[:32]}"

        return _fn

    researcher = A2AServer("研究员Agent", "负责资料检索与问题拆解", capabilities={"research": True, "summarize": True})
    writer = A2AServer("写作Agent", "负责结构化文案生成", capabilities={"draft": True, "rewrite": True})
    reviewer = A2AServer("审校Agent", "负责一致性检查与润色", capabilities={"review": True, "polish": True})

    researcher.add_skill("research", _skill_reply("研究结论"))
    researcher.add_skill("summarize", _skill_reply("摘要"))
    writer.add_skill("draft", _skill_reply("草稿"))
    writer.add_skill("rewrite", _skill_reply("改写"))
    reviewer.add_skill("review", _skill_reply("审校意见"))
    reviewer.add_skill("polish", _skill_reply("润色结果"))

    registry = AgentRegistry()
    registry.register_agent("研究员Agent", "http://localhost:8101", {"role": "research"})
    registry.register_agent("写作Agent", "http://localhost:8102", {"role": "writing"})
    registry.register_agent("审校Agent", "http://localhost:8103", {"role": "review"})

    network = AgentNetwork("终极协作网络")
    network.add_agent("研究员Agent", "http://localhost:8101")
    network.add_agent("写作Agent", "http://localhost:8102")
    network.add_agent("审校Agent", "http://localhost:8103")

    return {
        "source": "live",
        "agents": [researcher.get_info(), writer.get_info(), reviewer.get_info()],
        "registry": {"info": registry.get_info(), "agents": registry.list_agents()},
        "network": {"name": network.name, "agents": network.list_agents()},
        "delegation_flow": [
            {"from": "用户", "to": "研究员Agent", "message": "请调研终极 dashboard 的设计方向。"},
            {"from": "研究员Agent", "to": "写作Agent", "message": "整理为模块化提纲并突出 live/demo 区分。"},
            {"from": "写作Agent", "to": "审校Agent", "message": "请检查中文文案、信息层级与术语一致性。"},
            {"from": "审校Agent", "to": "用户", "message": "已确认：侧边栏 + 9 面板 + ReAct SSE 需保留。"},
        ],
    }


def _anp_builder() -> Dict[str, Any]:
    discovery = ANPDiscovery()
    services = [
        ServiceInfo("svc-planner", "planning", "http://localhost:8201", "规划服务", ["task_decomposition", "routing"], {"zone": "core"}),
        ServiceInfo("svc-retrieval", "retrieval", "http://localhost:8202", "检索服务", ["vector_search", "memory_lookup"], {"zone": "core"}),
        ServiceInfo("svc-vision", "vision", "http://localhost:8203", "视觉服务", ["image_understanding"], {"zone": "perception"}),
        ServiceInfo("svc-nlp", "nlp", "http://localhost:8204", "语言服务", ["summarize", "extract"], {"zone": "core"}),
        ServiceInfo("svc-tooling", "tooling", "http://localhost:8205", "工具服务", ["calculate", "browser"], {"zone": "tools"}),
    ]
    for service in services:
        register_service(discovery, service=service)

    by_type = {stype: [s.to_dict() for s in discover_service(discovery, service_type=stype)] for stype in ["planning", "retrieval", "vision", "nlp", "tooling"]}

    network = ANPNetwork("ultimate-anp")
    for service in services:
        network.add_node(service.service_id, service.endpoint, {"service_type": service.service_type, **service.metadata})

    network.connect_nodes("svc-planner", "svc-retrieval")
    network.connect_nodes("svc-planner", "svc-nlp")
    network.connect_nodes("svc-planner", "svc-tooling")
    network.connect_nodes("svc-retrieval", "svc-nlp")
    network.connect_nodes("svc-vision", "svc-tooling")
    network.connect_nodes("svc-tooling", "svc-planner")

    route = network.route_message("svc-vision", "svc-nlp", {"task": "describe"})
    broadcast = network.broadcast_message("svc-planner", {"task": "sync"})
    stats = network.get_network_stats()

    edges = []
    for src, dsts in [
        ("svc-planner", ["svc-retrieval", "svc-nlp", "svc-tooling"]),
        ("svc-retrieval", ["svc-nlp"]),
        ("svc-vision", ["svc-tooling"]),
        ("svc-tooling", ["svc-planner"]),
    ]:
        for dst in dsts:
            edges.append({"from": src, "to": dst})

    return {
        "source": "live",
        "services": [s.to_dict() for s in services],
        "discovery_by_type": by_type,
        "network": {
            "network_id": network.network_id,
            "nodes": [network.get_node_info(s.service_id) for s in services],
            "edges": edges,
        },
        "route_example": {"from": "svc-vision", "to": "svc-nlp", "path": route},
        "broadcast_example": {"from": "svc-planner", "recipients": broadcast},
        "stats": stats,
    }


def _rl_builder() -> Dict[str, Any]:
    TrainingConfig = RL_UTILS.TrainingConfig
    Reward = RL_REWARDS.MathRewardFunction()

    configs = {
        "SFT": TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            output_dir="./output/sft",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
        ).to_dict(),
        "GRPO": TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            output_dir="./output/grpo",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
        ).to_dict(),
        "PPO": TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            output_dir="./output/ppo",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
        ).to_dict(),
    }

    samples = [
        ("Let's solve it.\nFinal Answer: 2", "2"),
        ("We compute carefully.\nFinal Answer: 42", "42"),
        ("Reasoning...\nFinal Answer: 17", "17"),
        ("The answer is 15", "16"),
        ("Step 1\nStep 2\nFinal Answer: 9", "9"),
    ]
    completions = []
    rewards = []
    for completion, truth in samples:
        extracted = Reward.extract_answer(completion)
        reward = 1.0 if extracted and Reward.compare_answers(extracted, truth) else 0.0
        completions.append(
            {
                "completion": completion,
                "ground_truth": truth,
                "extracted_answer": extracted,
                "reward": reward,
            }
        )
        rewards.append(reward)

    aggregate = {
        "mean_reward": round(float(sum(rewards) / len(rewards)), 3),
        "max_reward": float(max(rewards)),
        "min_reward": float(min(rewards)),
        "accuracy": round(float(sum(1 for r in rewards if r > 0.5) / len(rewards)), 3),
        "num_samples": len(rewards),
    }

    curve = []
    value = 0.18
    for step in range(1, 31):
        value = min(0.98, value + 0.02 + math.sin(step / 3) * 0.01)
        curve.append({"step": step, "reward": round(max(0.0, min(1.0, value + (0.01 if step > 20 else 0.0))), 3)})

    return {
        "source": "demo",
        "configs": configs,
        "reward_examples": completions,
        "aggregate": aggregate,
        "curve": curve,
        "note": "RL 训练器依赖 torch/trl/datasets，当前面板使用高保真演示数据。",
    }


def _eval_builder() -> Dict[str, Any]:
    bfcl_metrics = BFCL.BFCLMetrics()
    bfcl_results = [
        {
            "id": "bfcl-001",
            "success": True,
            "score": 0.95,
            "execution_time": 1.12,
            "category": "simple_python",
            "predicted": [{"name": "calculate", "arguments": {"expression": "2+2"}}],
        },
        {
            "id": "bfcl-002",
            "success": True,
            "score": 0.82,
            "execution_time": 1.58,
            "category": "parallel_calls",
            "predicted": [{"name": "search", "arguments": {"query": "HelloAgents"}}, {"name": "calculate", "arguments": {"expression": "15*23+45"}}],
        },
        {
            "id": "bfcl-003",
            "success": False,
            "score": 0.31,
            "execution_time": 2.03,
            "category": "multi_step",
            "predicted": [{"name": "unknown", "arguments": {}}],
        },
        {
            "id": "bfcl-004",
            "success": True,
            "score": 0.76,
            "execution_time": 0.94,
            "category": "simple_python",
            "predicted": [{"name": "calculate", "arguments": {"expression": "390"}}],
        },
    ]
    bfcl = sanitize(bfcl_metrics.compute_metrics(bfcl_results))
    bfcl["source"] = "live"

    gaia_results = [
        {"exact_match": True, "partial_match": True, "level": 1, "score": 4.8, "execution_time": 15.2},
        {"exact_match": True, "partial_match": True, "level": 1, "score": 4.4, "execution_time": 11.9},
        {"exact_match": False, "partial_match": True, "level": 2, "score": 3.8, "execution_time": 24.6},
        {"exact_match": True, "partial_match": True, "level": 2, "score": 4.2, "execution_time": 18.1},
        {"exact_match": False, "partial_match": False, "level": 3, "score": 2.6, "execution_time": 39.7},
        {"exact_match": False, "partial_match": True, "level": 3, "score": 3.1, "execution_time": 33.3},
    ]
    gaia_metrics = sanitize(GAIA.GAIAMetrics().compute_metrics(gaia_results))
    gaia_metrics["source"] = "demo"

    win_rate = {
        "source": "demo",
        "metrics": {"win_rate": 0.58, "loss_rate": 0.27, "tie_rate": 0.15, "wins": 29, "losses": 14, "ties": 7, "total_comparisons": 50},
        "note": "Win Rate / LLM Judge 依赖外部 LLM 与网络，当前展示为高保真示例。",
    }

    llm_judge = {
        "source": "demo",
        "dimension_averages": {
            "helpfulness": 8.7,
            "accuracy": 8.4,
            "clarity": 8.9,
            "structure": 8.2,
        },
        "average_total_score": 8.55,
        "pass_rate": 0.92,
        "excellent_rate": 0.61,
    }

    return {
        "source": "mixed",
        "bfcl": bfcl,
        "gaia": gaia_metrics,
        "win_rate": win_rate,
        "llm_judge": llm_judge,
    }


def build_overview() -> Dict[str, Any]:
    context = _context_builder()
    mcp = _mcp_builder()
    a2a = _a2a_builder()
    anp = _anp_builder()
    rl = _rl_builder()
    evaluation = _eval_builder()
    memory = _memory_items()

    modules = [
        {
            "key": "overview",
            "title": "概览 Overview",
            "source": "live",
            "description": "汇总九大面板的状态、可用性与关键指标。",
            "metric": "9 个面板",
        },
        {
            "key": "react",
            "title": "ReAct 流程",
            "source": "live",
            "description": "保留现有 SSE 追踪流程，展示思考→行动→观察。",
            "metric": "8 事件",
        },
        {
            "key": "memory",
            "title": "记忆 Memory",
            "source": "demo",
            "description": "工作 / 情景 / 语义 / 感知记忆的统一示意。",
            "metric": f"{memory['stats']['count']} 条样本",
        },
        {
            "key": "context",
            "title": "上下文 Context",
            "source": "live",
            "description": "GSSC 流水线、历史压缩与截断机制演示。",
            "metric": f"{context['budget']['used_tokens']}/{context['budget']['available_tokens']} tokens",
        },
        {
            "key": "mcp",
            "title": "MCP",
            "source": "live",
            "description": "上下文对象的 messages/tools/resources/metadata 四段结构。",
            "metric": f"{len(mcp['built_context']['tools'])} tools",
        },
        {
            "key": "a2a",
            "title": "A2A",
            "source": "live",
            "description": "Agent 之间的注册、发现与任务委派。",
            "metric": f"{len(a2a['agents'])} agents",
        },
        {
            "key": "anp",
            "title": "ANP",
            "source": "live",
            "description": "服务发现、网络拓扑和消息路由。",
            "metric": f"{anp['stats']['total_nodes']} nodes / {anp['stats']['total_connections']} edges",
        },
        {
            "key": "rl",
            "title": "强化学习 RL",
            "source": "demo",
            "description": "SFT / GRPO / PPO 配置与奖励曲线。",
            "metric": f"{rl['aggregate']['accuracy']:.0%} 准确率",
        },
        {
            "key": "eval",
            "title": "评估 Evaluation",
            "source": "live",
            "description": "BFCL 实时指标 + GAIA / WinRate / LLM Judge 示意。",
            "metric": f"BFCL {evaluation['bfcl']['accuracy']:.0%}",
        },
    ]

    return {
        "source": "live",
        "version": __version__,
        "modules": modules,
    }


__all__ = [
    "sanitize",
    "build_overview",
    "_memory_items",
    "_context_builder",
    "_mcp_builder",
    "_a2a_builder",
    "_anp_builder",
    "_rl_builder",
    "_eval_builder",
]
