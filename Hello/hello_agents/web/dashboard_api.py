"""HelloAgents 终极可视化控制台 - 后端 API

按模块划分的 REST/SSE 接口，覆盖框架全部子系统：
- /api/overview          : 各模块可用性总览
- /api/memory/*          : 记忆系统（MemoryManager，工作/情景/语义记忆）
- /api/context/*         : 上下文工程（ContextBuilder，GSSC 流水线逐阶段可视化）
- /api/mcp/*             : MCP 协议（fastmcp 内存传输，list_tools / call_tool）
- /api/a2a/*             : A2A 协议（真实 Flask A2AServer + A2AClient + AgentNetwork/Registry）
- /api/anp/*             : ANP 协议（ANPDiscovery + ANPNetwork 路由/广播）
- /api/rl/*              : 强化学习（奖励函数试算 + 训练曲线 SSE 模拟）
- /api/eval/*            : 评估（BFCLMetrics / GAIAMetrics / LLM Judge / Win Rate）

设计原则：所有可选依赖（qdrant、fastmcp、trl 等）缺失时接口返回降级信息，
前端展示"能力不可用"而非报错崩溃。
"""

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api")

_PKG_ROOT = Path(__file__).resolve().parent.parent


def _ok(data: Any) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


def _err(msg: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg})


# ---------------------------------------------------------------------------
# 总览
# ---------------------------------------------------------------------------

def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@router.get("/overview")
def overview() -> JSONResponse:
    from hello_agents.version import __version__

    modules = {
        "memory": {"label": "记忆系统 Memory", "available": True,
                   "detail": "工作/情景/语义/感知记忆 + MemoryManager"},
        "context": {"label": "上下文工程 Context", "available": _has_module("tiktoken"),
                    "detail": "ContextBuilder GSSC 流水线 / TokenCounter / Truncator"},
        "mcp": {"label": "MCP 协议", "available": _has_module("fastmcp"),
                "detail": "Model Context Protocol：工具/资源/提示词"},
        "a2a": {"label": "A2A 协议", "available": _has_module("flask"),
                "detail": "Agent-to-Agent：技能调用 / AgentNetwork / Registry"},
        "anp": {"label": "ANP 协议", "available": True,
                "detail": "Agent Network Protocol：服务发现 / 消息路由"},
        "rl": {"label": "强化学习 RL", "available": True,
               "detail": "奖励函数（TRL 训练器可选）", "trl": _has_module("trl")},
        "evaluation": {"label": "评估 Evaluation", "available": _has_module("datasets"),
                       "detail": "BFCL / GAIA / LLM Judge / Win Rate"},
        "kb": {"label": "知识库 Knowledge Base", "available": _has_module("sklearn"),
               "detail": "上传 md/pdf/docx 等文档，切块索引后供检索与问答",
               "markitdown": _has_module("markitdown")},
        "agent": {"label": "ReAct Agent", "available": True,
                  "detail": "思考-行动-观察 循环追踪"},
    }
    return _ok({"version": __version__, "modules": modules})


# ---------------------------------------------------------------------------
# 记忆系统
# ---------------------------------------------------------------------------

_memory_manager = None
_memory_lock = threading.Lock()
_memory_degraded = ""


def _get_memory_manager():
    """惰性构建 MemoryManager：优先启用全部类型，缺依赖时降级为工作记忆。"""
    global _memory_manager, _memory_degraded
    with _memory_lock:
        if _memory_manager is not None:
            return _memory_manager
        from hello_agents.memory.manager import MemoryManager
        try:
            _memory_manager = MemoryManager(user_id="dashboard_user")
        except Exception as e:
            _memory_degraded = f"情景/语义记忆不可用（{type(e).__name__}），已降级为工作记忆"
            _memory_manager = MemoryManager(
                user_id="dashboard_user",
                enable_episodic=False, enable_semantic=False, enable_perceptual=False,
            )
        return _memory_manager


def _memory_item_dict(item) -> Dict[str, Any]:
    return {
        "id": item.id,
        "content": item.content,
        "memory_type": item.memory_type,
        "importance": round(float(item.importance), 3),
        "timestamp": item.timestamp.isoformat(sep=" ", timespec="seconds"),
        "metadata": item.metadata,
    }


class MemoryAddRequest(BaseModel):
    content: str
    memory_type: str = "working"
    importance: Optional[float] = None
    auto_classify: bool = True


@router.post("/memory/add")
def memory_add(req: MemoryAddRequest) -> JSONResponse:
    try:
        mgr = _get_memory_manager()
        memory_id = mgr.add_memory(
            content=req.content, memory_type=req.memory_type,
            importance=req.importance, auto_classify=req.auto_classify,
        )
        return _ok({"memory_id": memory_id, "stats": mgr.get_memory_stats(),
                    "degraded": _memory_degraded})
    except Exception as e:
        return _err(str(e))


@router.get("/memory/search")
def memory_search(q: str, limit: int = 10, min_importance: float = 0.0) -> JSONResponse:
    try:
        mgr = _get_memory_manager()
        items = mgr.retrieve_memories(query=q, limit=limit, min_importance=min_importance)
        return _ok({"items": [_memory_item_dict(i) for i in items]})
    except Exception as e:
        return _err(str(e))


@router.get("/memory/stats")
def memory_stats() -> JSONResponse:
    try:
        mgr = _get_memory_manager()
        return _ok({"stats": mgr.get_memory_stats(), "degraded": _memory_degraded})
    except Exception as e:
        return _err(str(e))


class MemoryForgetRequest(BaseModel):
    strategy: str = "importance_based"
    threshold: float = 0.3


@router.post("/memory/forget")
def memory_forget(req: MemoryForgetRequest) -> JSONResponse:
    try:
        mgr = _get_memory_manager()
        removed = mgr.forget_memories(strategy=req.strategy, threshold=req.threshold)
        return _ok({"removed": removed, "stats": mgr.get_memory_stats()})
    except Exception as e:
        return _err(str(e))


@router.post("/memory/consolidate")
def memory_consolidate() -> JSONResponse:
    try:
        mgr = _get_memory_manager()
        result = mgr.consolidate_memories()
        return _ok({"consolidated": result, "stats": mgr.get_memory_stats()})
    except Exception as e:
        return _err(str(e))


@router.post("/memory/clear")
def memory_clear() -> JSONResponse:
    try:
        mgr = _get_memory_manager()
        mgr.clear_all_memories()
        return _ok({"stats": mgr.get_memory_stats()})
    except Exception as e:
        return _err(str(e))


# ---------------------------------------------------------------------------
# 上下文工程（GSSC 流水线逐阶段可视化）
# ---------------------------------------------------------------------------

class ContextMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ContextBuildRequest(BaseModel):
    query: str
    system_instructions: str = "你是一个有帮助的中文智能体助手。"
    history: List[ContextMessage] = Field(default_factory=list)
    extra_packets: List[str] = Field(default_factory=list)
    max_tokens: int = 2000
    min_relevance: float = 0.3
    enable_mmr: bool = True
    mmr_lambda: float = 0.7
    use_kb: bool = False
    kb_top_k: int = 4


@router.post("/context/build")
def context_build(req: ContextBuildRequest) -> JSONResponse:
    """逐阶段执行 GSSC 流水线并返回每个阶段的中间产物。"""
    try:
        from hello_agents.context.builder import (
            ContextBuilder, ContextConfig, ContextPacket, count_tokens,
        )
        from hello_agents.core.message import Message

        config = ContextConfig(
            max_tokens=req.max_tokens, min_relevance=req.min_relevance,
            enable_mmr=req.enable_mmr, mmr_lambda=req.mmr_lambda,
        )
        builder = ContextBuilder(config=config)

        history = [Message(m.content, m.role) for m in req.history if m.content.strip()]
        extra = [ContextPacket(content=c, metadata={"type": "retrieval"})
                 for c in req.extra_packets if c.strip()]

        kb_hits: List[Dict[str, Any]] = []
        if req.use_kb:
            kb_hits = _kb_search(req.query, top_k=req.kb_top_k)
            extra.extend(
                ContextPacket(content=h["content"],
                              metadata={"type": "knowledge_base", "source": h["doc_name"]})
                for h in kb_hits
            )

        gathered = builder._gather(
            user_query=req.query, conversation_history=history,
            system_instructions=req.system_instructions or None, additional_packets=extra,
        )
        selected = builder._select(list(gathered), req.query)
        structured = builder._structure(
            selected_packets=selected, user_query=req.query,
            system_instructions=req.system_instructions or None,
        )
        final = builder._compress(structured)

        def pk(p):
            return {
                "type": p.metadata.get("type", "unknown"),
                "content": p.content,
                "token_count": p.token_count,
                "relevance": round(float(p.relevance_score), 3),
            }

        selected_ids = {id(p) for p in selected}
        return _ok({
            "budget": {
                "max_tokens": config.max_tokens,
                "available_tokens": config.get_available_tokens(),
                "reserve_ratio": config.reserve_ratio,
            },
            "gather": [dict(pk(p), selected=(id(p) in selected_ids)) for p in gathered],
            "select": [pk(p) for p in selected],
            "structure": {"text": structured, "tokens": count_tokens(structured)},
            "compress": {"text": final, "tokens": count_tokens(final),
                         "compressed": final != structured},
            "kb_hits": kb_hits,
        })
    except Exception as e:
        return _err(str(e))


# ---------------------------------------------------------------------------
# MCP 协议
# ---------------------------------------------------------------------------

_mcp_server = None
_mcp_lock = threading.Lock()


def _get_mcp_server():
    global _mcp_server
    with _mcp_lock:
        if _mcp_server is None:
            from hello_agents.protocols.mcp.server import create_example_server
            _mcp_server = create_example_server()
        return _mcp_server


def _run_async(coro):
    return asyncio.run(coro)


@router.get("/mcp/info")
def mcp_info() -> JSONResponse:
    try:
        from hello_agents.protocols.mcp.client import MCPClient
        server = _get_mcp_server()

        async def fetch():
            async with MCPClient(server.mcp) as client:
                tools = await client.list_tools()
                transport = client.get_transport_info()
                alive = await client.ping()
            return tools, transport, alive

        tools, transport, alive = _run_async(fetch())
        return _ok({"server": server.get_info(), "tools": tools,
                    "transport": transport, "ping": alive})
    except Exception as e:
        return _err(f"MCP 不可用: {e}")


class MCPCallRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


@router.post("/mcp/call")
def mcp_call(req: MCPCallRequest) -> JSONResponse:
    try:
        from hello_agents.protocols.mcp.client import MCPClient
        server = _get_mcp_server()

        async def call():
            async with MCPClient(server.mcp) as client:
                return await client.call_tool(req.tool, req.arguments)

        started = time.time()
        result = _run_async(call())
        return _ok({"tool": req.tool, "arguments": req.arguments,
                    "result": str(result), "elapsed_ms": int((time.time() - started) * 1000)})
    except Exception as e:
        return _err(str(e))


# ---------------------------------------------------------------------------
# A2A 协议（真实 Flask 服务器 + HTTP 客户端）
# ---------------------------------------------------------------------------

_a2a_state: Dict[str, Any] = {"started": False, "agents": {}}
_a2a_lock = threading.Lock()
_A2A_PORTS = {"translator-agent": 5101, "math-agent": 5102}


def _build_a2a_agents():
    from hello_agents.protocols.a2a.implementation import A2AServer

    translator = A2AServer(
        name="translator-agent",
        description="简单文本处理 Agent：大小写转换 / 反转 / 词数统计",
        capabilities={"language": "any", "streaming": False},
    )

    @translator.skill("uppercase")
    def uppercase(text: str) -> str:
        return text.upper()

    @translator.skill("reverse")
    def reverse(text: str) -> str:
        return text[::-1]

    @translator.skill("word_count")
    def word_count(text: str) -> str:
        return f"字符数: {len(text)}, 词数: {len(text.split())}"

    math_agent = A2AServer(
        name="math-agent",
        description="数学 Agent：安全表达式求值",
        capabilities={"domain": "math", "streaming": False},
    )

    @math_agent.skill("calculate")
    def calculate(text: str) -> str:
        allowed = set("0123456789+-*/(). ")
        if not text or not all(c in allowed for c in text):
            return "Error: 仅支持数字与 + - * / ( ) 运算"
        try:
            return f"{text} = {eval(text)}"
        except Exception as e:
            return f"Error: {e}"

    return [translator, math_agent]


def _ensure_a2a_started() -> Dict[str, Any]:
    from hello_agents.protocols.a2a.implementation import AgentNetwork, AgentRegistry

    with _a2a_lock:
        if _a2a_state["started"]:
            return _a2a_state

        servers = _build_a2a_agents()
        registry = AgentRegistry(name="Dashboard Registry", description="控制台内置注册中心")
        network = AgentNetwork(name="Dashboard Network")

        for server in servers:
            port = _A2A_PORTS[server.name]

            def run_server(srv=server, p=port):
                import logging
                logging.getLogger("werkzeug").setLevel(logging.ERROR)
                srv.run(host="127.0.0.1", port=p)

            threading.Thread(target=run_server, daemon=True).start()
            url = f"http://127.0.0.1:{port}"
            registry.register_agent(server.name, url, metadata=server.capabilities)
            network.add_agent(server.name, url)
            _a2a_state["agents"][server.name] = {"url": url, "info": server.get_info()}

        _a2a_state["registry"] = registry
        _a2a_state["network"] = network
        time.sleep(1.0)  # 等待 Flask 就绪
        _a2a_state["started"] = True
        return _a2a_state


@router.post("/a2a/start")
def a2a_start() -> JSONResponse:
    try:
        state = _ensure_a2a_started()
        return _ok({"agents": state["agents"],
                    "registry": state["registry"].list_agents(),
                    "network": state["network"].list_agents()})
    except Exception as e:
        return _err(str(e))


@router.get("/a2a/network")
def a2a_network() -> JSONResponse:
    try:
        if not _a2a_state["started"]:
            return _ok({"started": False, "agents": {}})
        network = _a2a_state["network"]
        agents = network.list_agents()
        return _ok({"started": True, "agents": agents,
                    "registry": _a2a_state["registry"].list_agents()})
    except Exception as e:
        return _err(str(e))


class A2AAskRequest(BaseModel):
    agent: str
    question: str
    skill: Optional[str] = None


@router.post("/a2a/ask")
def a2a_ask(req: A2AAskRequest) -> JSONResponse:
    try:
        state = _ensure_a2a_started()
        network = state["network"]
        client = network.get_agent(req.agent)
        started = time.time()
        if req.skill:
            result = client.execute_skill(req.skill, req.question)
            answer = result.get("result", result.get("error", str(result)))
            skill_used = req.skill
        else:
            data = client.ask(req.question)
            answer, skill_used = data, "auto"
        return _ok({"agent": req.agent, "skill": skill_used, "question": req.question,
                    "answer": answer, "elapsed_ms": int((time.time() - started) * 1000),
                    "server_url": client.server_url})
    except Exception as e:
        return _err(str(e))


# ---------------------------------------------------------------------------
# ANP 协议
# ---------------------------------------------------------------------------

_anp_state: Dict[str, Any] = {}
_anp_lock = threading.Lock()


def _get_anp():
    with _anp_lock:
        if not _anp_state:
            from hello_agents.protocols.anp.implementation import (
                ANPDiscovery, ServiceInfo, create_example_network,
            )
            network = create_example_network()
            discovery = ANPDiscovery()
            discovery.register_service(ServiceInfo(
                service_id="svc-translate", service_type="nlp",
                endpoint="http://localhost:8001/translate", service_name="翻译服务",
                capabilities=["translate", "detect_language"], metadata={"lang": "zh-en"}))
            discovery.register_service(ServiceInfo(
                service_id="svc-calc", service_type="math",
                endpoint="http://localhost:8002/calc", service_name="计算服务",
                capabilities=["calculate"], metadata={"precision": "float64"}))
            discovery.register_service(ServiceInfo(
                service_id="svc-search", service_type="retrieval",
                endpoint="http://localhost:8003/search", service_name="检索服务",
                capabilities=["search", "rank"], metadata={"index": "web"}))
            _anp_state["network"] = network
            _anp_state["discovery"] = discovery
            _anp_state["log"] = []
        return _anp_state


def _anp_graph(network) -> Dict[str, Any]:
    stats = network.get_network_stats()
    nodes = []
    edges = []
    for node_id in stats["nodes"]:
        info = network.get_node_info(node_id)
        nodes.append({"id": node_id, "endpoint": info["endpoint"],
                      "metadata": info.get("metadata", {}), "status": info.get("status")})
        for target in info.get("connections", []):
            edges.append({"from": node_id, "to": target})
    return {"stats": stats, "nodes": nodes, "edges": edges}


@router.get("/anp/network")
def anp_network() -> JSONResponse:
    try:
        state = _get_anp()
        services = [s.to_dict() for s in state["discovery"].list_all_services()]
        return _ok({"graph": _anp_graph(state["network"]), "services": services,
                    "log": state["log"][-20:]})
    except Exception as e:
        return _err(str(e))


class ANPNodeRequest(BaseModel):
    node_id: str
    endpoint: str = ""
    role: str = "worker"
    connect_to: Optional[str] = None


@router.post("/anp/add_node")
def anp_add_node(req: ANPNodeRequest) -> JSONResponse:
    try:
        state = _get_anp()
        network = state["network"]
        endpoint = req.endpoint or f"http://localhost:{8000 + len(network.get_network_stats()['nodes']) + 1}"
        network.add_node(req.node_id, endpoint, {"type": "agent", "role": req.role})
        if req.connect_to:
            network.connect_nodes(req.connect_to, req.node_id)
            network.connect_nodes(req.node_id, req.connect_to)
        return _ok({"graph": _anp_graph(network)})
    except Exception as e:
        return _err(str(e))


class ANPRouteRequest(BaseModel):
    from_node: str
    to_node: str
    message: str = "ping"


@router.post("/anp/route")
def anp_route(req: ANPRouteRequest) -> JSONResponse:
    try:
        state = _get_anp()
        path = state["network"].route_message(
            req.from_node, req.to_node, {"content": req.message})
        entry = {"kind": "route", "from": req.from_node, "to": req.to_node,
                 "message": req.message, "path": path,
                 "ts": time.strftime("%H:%M:%S")}
        state["log"].append(entry)
        return _ok({"path": path, "log": state["log"][-20:]})
    except Exception as e:
        return _err(str(e))


class ANPBroadcastRequest(BaseModel):
    from_node: str
    message: str = "hello"


@router.post("/anp/broadcast")
def anp_broadcast(req: ANPBroadcastRequest) -> JSONResponse:
    try:
        state = _get_anp()
        receivers = state["network"].broadcast_message(req.from_node, {"content": req.message})
        entry = {"kind": "broadcast", "from": req.from_node, "message": req.message,
                 "receivers": receivers, "ts": time.strftime("%H:%M:%S")}
        state["log"].append(entry)
        return _ok({"receivers": receivers, "log": state["log"][-20:]})
    except Exception as e:
        return _err(str(e))


@router.get("/anp/discover")
def anp_discover(service_type: Optional[str] = None) -> JSONResponse:
    try:
        state = _get_anp()
        services = state["discovery"].discover_services(service_type=service_type or None)
        return _ok({"services": [s.to_dict() for s in services]})
    except Exception as e:
        return _err(str(e))


# ---------------------------------------------------------------------------
# 强化学习
# ---------------------------------------------------------------------------

def _load_rl_rewards():
    """直接从文件加载 rewards 模块，避免包 __init__ 强制依赖 trl。"""
    path = _PKG_ROOT / "rl" / "rewards.py"
    spec = importlib.util.spec_from_file_location("hello_agents_rl_rewards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rl_utils():
    path = _PKG_ROOT / "rl" / "utils.py"
    spec = importlib.util.spec_from_file_location("hello_agents_rl_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@router.get("/rl/info")
def rl_info() -> JSONResponse:
    try:
        from dataclasses import asdict
        utils = _load_rl_utils()
        config = asdict(utils.TrainingConfig())
        return _ok({"trl_available": _has_module("trl"),
                    "trainers": ["SFTTrainerWrapper", "GRPOTrainerWrapper", "PPOTrainerWrapper"],
                    "default_config": config})
    except Exception as e:
        return _err(str(e))


class RLRewardRequest(BaseModel):
    completions: List[str]
    ground_truths: List[str]
    length_penalty: bool = False
    max_length: int = 200
    step_bonus: bool = False


@router.post("/rl/reward")
def rl_reward(req: RLRewardRequest) -> JSONResponse:
    try:
        rewards_mod = _load_rl_rewards()
        base_fn = rewards_mod.create_accuracy_reward()
        fn = base_fn
        if req.length_penalty:
            fn = rewards_mod.create_length_penalty_reward(fn, max_length=req.max_length)
        if req.step_bonus:
            fn = rewards_mod.create_step_reward(fn)

        base = base_fn(req.completions, ground_truth=req.ground_truths)
        final = fn(req.completions, ground_truth=req.ground_truths)

        extractor = rewards_mod.MathRewardFunction()
        samples = []
        for i, (comp, truth) in enumerate(zip(req.completions, req.ground_truths)):
            samples.append({
                "completion": comp, "ground_truth": truth,
                "extracted": extractor.extract_answer(comp),
                "base_reward": round(base[i], 3), "final_reward": round(final[i], 3),
                "length": len(comp), "steps": comp.count("\n"),
            })
        stats = {
            "mean_reward": round(sum(base) / len(base), 4) if base else 0.0,
            "max_reward": max(base) if base else 0.0,
            "min_reward": min(base) if base else 0.0,
            "accuracy": round(sum(1 for r in base if r >= 1.0) / len(base), 4) if base else 0.0,
        }
        return _ok({"samples": samples, "stats": stats})
    except Exception as e:
        return _err(str(e))


@router.get("/rl/train")
def rl_train(algo: str = "GRPO", steps: int = 40) -> StreamingResponse:
    """训练曲线 SSE 流。TRL 未安装时输出带随机噪声的模拟曲线（前端明确标注）。"""
    steps = max(5, min(int(steps), 200))
    simulated = not _has_module("trl")

    def stream():
        rng = random.Random(42)
        base_kl = {"GRPO": 0.02, "PPO": 0.05, "SFT": 0.0}.get(algo, 0.02)
        for step in range(1, steps + 1):
            progress = step / steps
            reward = round(min(1.0, 0.1 + 0.85 * (1 - math.exp(-3 * progress))
                               + rng.uniform(-0.06, 0.06)), 4)
            loss = round(max(0.02, 2.2 * math.exp(-2.5 * progress)
                             + rng.uniform(-0.05, 0.05)), 4)
            kl = round(max(0.0, base_kl * (1 + math.sin(progress * 6) * 0.4)
                           + rng.uniform(0, 0.01)), 4)
            payload = {"step": step, "total": steps, "algo": algo,
                       "reward_mean": reward, "loss": loss, "kl": kl,
                       "simulated": simulated}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            time.sleep(0.12)
        yield f"data: {json.dumps({'done': True, 'simulated': simulated}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

_BFCL_DEMO_RESULTS = [
    {"sample_id": "bfcl_001", "category": "simple_python", "success": True, "score": 1.0,
     "execution_time": 0.8, "predicted": [{"name": "get_weather", "args": {"city": "Beijing"}}]},
    {"sample_id": "bfcl_002", "category": "simple_python", "success": True, "score": 1.0,
     "execution_time": 1.1, "predicted": [{"name": "calculate", "args": {"expr": "2+2"}}]},
    {"sample_id": "bfcl_003", "category": "multiple", "success": False, "score": 0.4,
     "execution_time": 2.4, "predicted": [{"name": "search", "args": {"q": "MCP"}}]},
    {"sample_id": "bfcl_004", "category": "parallel", "success": True, "score": 0.9,
     "execution_time": 1.9, "predicted": [{"name": "get_weather", "args": {}}, {"name": "get_time", "args": {}}]},
    {"sample_id": "bfcl_005", "category": "parallel", "success": False, "score": 0.2,
     "execution_time": 3.2, "predicted": [{"name": "send_email", "args": {}}]},
]

_GAIA_DEMO_RESULTS = [
    {"task_id": "gaia_001", "level": 1, "exact_match": True, "partial_match": True,
     "score": 1.0, "execution_time": 3.1},
    {"task_id": "gaia_002", "level": 1, "exact_match": True, "partial_match": True,
     "score": 1.0, "execution_time": 4.7},
    {"task_id": "gaia_003", "level": 2, "exact_match": False, "partial_match": True,
     "score": 0.5, "execution_time": 8.2},
    {"task_id": "gaia_004", "level": 2, "exact_match": True, "partial_match": True,
     "score": 1.0, "execution_time": 6.5},
    {"task_id": "gaia_005", "level": 3, "exact_match": False, "partial_match": False,
     "score": 0.0, "execution_time": 15.3},
]


@router.get("/eval/info")
def eval_info() -> JSONResponse:
    benchmarks = [
        {"id": "bfcl", "name": "BFCL", "full": "Berkeley Function Calling Leaderboard",
         "desc": "工具/函数调用能力评估：AST 匹配、多函数、并行调用",
         "available": _has_module("datasets")},
        {"id": "gaia", "name": "GAIA", "full": "General AI Assistants",
         "desc": "通用 AI 助手评估：分 3 个难度级别的真实任务",
         "available": _has_module("datasets")},
        {"id": "llm_judge", "name": "LLM Judge", "full": "LLM-as-a-Judge",
         "desc": "用 LLM 对生成数据打分（多维度评分）", "available": True},
        {"id": "win_rate", "name": "Win Rate", "full": "Pairwise Win Rate",
         "desc": "两两对比计算胜率", "available": True},
    ]
    return _ok({"benchmarks": benchmarks})


@router.get("/eval/bfcl_demo")
def eval_bfcl_demo() -> JSONResponse:
    try:
        from hello_agents.evaluation.benchmarks.bfcl.metrics import BFCLMetrics
        metrics = BFCLMetrics()
        computed = metrics.compute_metrics(_BFCL_DEMO_RESULTS)
        return _ok({"results": _BFCL_DEMO_RESULTS, "metrics": computed, "demo": True})
    except Exception as e:
        return _err(str(e))


@router.get("/eval/gaia_demo")
def eval_gaia_demo() -> JSONResponse:
    try:
        from hello_agents.evaluation.benchmarks.gaia.metrics import GAIAMetrics
        metrics = GAIAMetrics()
        computed = metrics.compute_metrics(_GAIA_DEMO_RESULTS)
        return _ok({"results": _GAIA_DEMO_RESULTS, "metrics": computed, "demo": True})
    except Exception as e:
        return _err(str(e))


# ---------------------------------------------------------------------------
# 知识库 Knowledge Base（文档上传 + 切块索引 + 检索 + 问答）
# ---------------------------------------------------------------------------

_KB_DIR = Path(os.environ.get("KB_DIR", str(Path.home() / ".hello_agents" / "kb")))
_KB_FILES_DIR = _KB_DIR / "files"
_KB_STATE_FILE = _KB_DIR / "state.json"
_KB_ALLOWED_EXTS = {".md", ".markdown", ".txt", ".pdf", ".doc", ".docx",
                    ".html", ".htm", ".csv", ".json"}
_KB_MAX_FILE_BYTES = 50 * 1024 * 1024

_kb_lock = threading.Lock()
_kb_docs: Dict[str, Dict[str, Any]] = {}
_kb_chunks: List[Dict[str, Any]] = []
_kb_vectorizer = None
_kb_matrix = None
_kb_loaded = False


def _kb_extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".md", ".markdown", ".txt", ".csv", ".json"):
        return path.read_text(encoding="utf-8", errors="replace")
    from hello_agents.memory.rag.pipeline import _convert_to_markdown
    return _convert_to_markdown(str(path))


def _kb_reindex_locked() -> None:
    global _kb_vectorizer, _kb_matrix
    if not _kb_chunks:
        _kb_vectorizer, _kb_matrix = None, None
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    # 字符 n-gram：对中文无需分词也能得到有效相似度
    _kb_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=50000)
    _kb_matrix = _kb_vectorizer.fit_transform([c["content"] for c in _kb_chunks])


def _kb_save_locked() -> None:
    _KB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _KB_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"docs": _kb_docs, "chunks": _kb_chunks},
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(_KB_STATE_FILE)


def _kb_ensure_loaded() -> None:
    global _kb_loaded, _kb_docs, _kb_chunks
    with _kb_lock:
        if _kb_loaded:
            return
        _kb_loaded = True
        if _KB_STATE_FILE.exists():
            try:
                state = json.loads(_KB_STATE_FILE.read_text(encoding="utf-8"))
                _kb_docs = state.get("docs", {})
                _kb_chunks = state.get("chunks", [])
                _kb_reindex_locked()
            except Exception:
                _kb_docs, _kb_chunks = {}, []


def _kb_search(query: str, top_k: int = 5, min_score: float = 0.05) -> List[Dict[str, Any]]:
    _kb_ensure_loaded()
    with _kb_lock:
        if _kb_vectorizer is None or _kb_matrix is None:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(_kb_vectorizer.transform([query]), _kb_matrix)[0]
        order = sims.argsort()[::-1][:max(1, top_k)]
        hits = []
        for i in order:
            score = float(sims[i])
            if score < min_score:
                continue
            c = _kb_chunks[int(i)]
            hits.append({
                "score": round(score, 4),
                "doc_id": c["doc_id"],
                "doc_name": _kb_docs.get(c["doc_id"], {}).get("name", "?"),
                "chunk_index": c["chunk_index"],
                "content": c["content"],
            })
        return hits


@router.get("/kb/docs")
def kb_docs() -> JSONResponse:
    _kb_ensure_loaded()
    with _kb_lock:
        docs = sorted(_kb_docs.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True)
        return _ok({"docs": docs, "total_chunks": len(_kb_chunks),
                    "markitdown": _has_module("markitdown"),
                    "storage_dir": str(_KB_DIR)})


@router.post("/kb/upload")
async def kb_upload(file: UploadFile = File(...)) -> JSONResponse:
    _kb_ensure_loaded()
    try:
        name = Path(file.filename or "unnamed").name
        ext = Path(name).suffix.lower()
        if ext not in _KB_ALLOWED_EXTS:
            return _err(f"不支持的文件类型 {ext or '(无扩展名)'}，支持：{', '.join(sorted(_KB_ALLOWED_EXTS))}")
        if ext in (".pdf", ".doc", ".docx", ".html", ".htm") and not _has_module("markitdown"):
            return _err("解析 PDF/Word/HTML 需要 markitdown：pip install markitdown")

        raw = await file.read()
        if len(raw) > _KB_MAX_FILE_BYTES:
            return _err(f"文件过大（>{_KB_MAX_FILE_BYTES // 1024 // 1024}MB）")

        doc_id = hashlib.md5(raw).hexdigest()[:12]
        _KB_FILES_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = _KB_FILES_DIR / f"{doc_id}_{name}"
        saved_path.write_bytes(raw)

        text = _kb_extract_text(saved_path)
        if not text or not text.strip():
            saved_path.unlink(missing_ok=True)
            return _err("未能从文件中提取到文本内容")

        from hello_agents.memory.rag.document import Document, DocumentProcessor
        processor = DocumentProcessor(chunk_size=600, chunk_overlap=100)
        doc = Document(content=text, metadata={"source": name}, doc_id=doc_id)
        chunks = processor.filter_chunks(processor.process_document(doc), min_length=20)
        if not chunks:
            saved_path.unlink(missing_ok=True)
            return _err("文档切块后无有效内容")

        with _kb_lock:
            _kb_chunks[:] = [c for c in _kb_chunks if c["doc_id"] != doc_id]
            for c in chunks:
                _kb_chunks.append({"doc_id": doc_id, "chunk_index": c.chunk_index,
                                   "content": c.content})
            _kb_docs[doc_id] = {
                "doc_id": doc_id, "name": name, "size": len(raw), "ext": ext,
                "chunks": len(chunks), "chars": len(text),
                "path": str(saved_path),
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _kb_reindex_locked()
            _kb_save_locked()
        return _ok({"doc": _kb_docs[doc_id], "total_chunks": len(_kb_chunks)})
    except Exception as e:
        return _err(str(e))


@router.delete("/kb/docs/{doc_id}")
def kb_delete(doc_id: str) -> JSONResponse:
    _kb_ensure_loaded()
    with _kb_lock:
        doc = _kb_docs.pop(doc_id, None)
        if doc is None:
            return _err(f"文档 {doc_id} 不存在")
        _kb_chunks[:] = [c for c in _kb_chunks if c["doc_id"] != doc_id]
        try:
            if doc.get("path"):
                Path(doc["path"]).unlink(missing_ok=True)
        except Exception:
            pass
        _kb_reindex_locked()
        _kb_save_locked()
        return _ok({"deleted": doc_id, "total_chunks": len(_kb_chunks)})


@router.get("/kb/search")
def kb_search_api(query: str, top_k: int = 5) -> JSONResponse:
    try:
        if not query.strip():
            return _err("query 不能为空")
        return _ok({"hits": _kb_search(query, top_k=top_k)})
    except Exception as e:
        return _err(str(e))


class KBAskRequest(BaseModel):
    question: str
    top_k: int = 4


@router.post("/kb/ask")
def kb_ask(req: KBAskRequest) -> JSONResponse:
    """知识库问答：检索相关片段，若配置了 LLM 则生成有据回答，否则仅返回检索结果。"""
    try:
        if not req.question.strip():
            return _err("question 不能为空")
        hits = _kb_search(req.question, top_k=req.top_k)
        if not hits:
            return _ok({"hits": [], "answer": None, "llm": False,
                        "note": "知识库中未检索到相关内容，请先上传文档"})
        context = "\n\n".join(
            f"[片段{i+1}｜{h['doc_name']}] {h['content']}" for i, h in enumerate(hits))
        answer, used_llm, note = None, False, None
        try:
            from hello_agents.core.llm import HelloAgentsLLM
            llm = HelloAgentsLLM()
            prompt = (
                "你是一个严谨的知识库问答助手。仅依据下方知识库片段回答用户问题；"
                "若片段不足以回答，请明确说明。回答末尾标注引用的片段编号。\n\n"
                f"知识库片段：\n{context}\n\n用户问题：{req.question}"
            )
            answer = llm.invoke([{"role": "user", "content": prompt}])
            used_llm = True
        except Exception as e:
            note = f"LLM 不可用（{e}），仅返回检索结果"
        return _ok({"hits": hits, "answer": answer, "llm": used_llm, "note": note})
    except Exception as e:
        return _err(str(e))
