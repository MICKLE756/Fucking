"""
专利检索动态工作流 - 节点 + 路由 + 状态管理

意图识别流水线: 规则 → 远程微调模型 → OpenAI 兜底 → 状态机修正
专利检索: 基于 milvus.json 的关键词匹配检索

工作流图:
  confirm_check → (affirm)         → tool_execute → response_gen → END
                → (not_confirming) → intent_recognize
                     → (chitchat)       → chitchat_reply → END
                     → (feedback)       → feedback_handle → entity_extract → completeness_eval → ...
                     → (direct_execute) → entity_extract → tool_execute → response_gen → END
                     → (need_eval)      → entity_extract → completeness_eval
                                             → (incomplete) → clarify_gen → END
                                             → (complete)   → confirm_gen → END
"""

import json
import logging

import config
import prompts
from dialog_llm import DialogLLM, GROUNDING_PROMPT
from intent_recognize_rule_base import IntentRecognizeRuleBase
from intent_recognize_model_base import IntentRecognizeModelBase
from intent_state_machine import IntentStateMachine
from entity_extractor_rule_base import EntityExtractorRuleBase
from intent_slot_memory import SlotMemory
from query_history_memory import QueryHistoryMemory
from patent_search import PatentSearchService
from workflow_engine import WorkflowEngine

logger = logging.getLogger(__name__)


class PatentWorkflow:
    """专利检索动态工作流"""

    # 规则层置信度阈值：低于 MIN_SCORE 或 Top1/Top2 分差低于 MIN_MARGIN 视为「模糊」，
    # 不直接采用规则结果，而是降级到远程模型 / OpenAI 兜底，避免子串误命中一票否决。
    RULE_MIN_SCORE = 2.0
    RULE_MIN_MARGIN = 1.0

    def __init__(self):
        self.llm = DialogLLM()
        self.intent_recognizer = IntentRecognizeRuleBase()
        self.intent_model = IntentRecognizeModelBase()
        self.intent_sm = IntentStateMachine()
        self.entity_extractor = EntityExtractorRuleBase()
        self.slot_memory = SlotMemory()
        self.query_history = QueryHistoryMemory()
        self.search_service = PatentSearchService()
        self.state = self._default_state()
        self.engine = self._build_engine()

    # ==================== 状态管理 ====================

    @staticmethod
    def _default_state() -> dict:
        return {
            "user_input": "",
            "session_id": "",
            "phase": "idle",
            "intent": {},
            "intent_level1": "",
            "intent_level2": [],
            "tech_domain": "",
            "core_problem": "",
            "constraints": {},
            "filters": {},
            "clarify_round": 0,
            "max_clarify_round": 3,
            "confirmed": False,
            "history": [],
            # 内部路由标记
            "_post_extract_route": "",   # "direct_execute" | "eval"
            "_pending_slot": "",         # 上一轮追问的槽位（用于回填学习）
            "_pending_intent": (),       # 发起该追问的 (一级, 二级) 意图
            "_gather_query": "",         # 触发本段信息收集的原始问题（用于沉淀示例）
            "_response": "",
            "_request": "",
        }

    # 槽位名 → 该槽位是否已填的判断依据（state 字段）
    @staticmethod
    def _slot_filled(state: dict, slot: str) -> bool:
        if slot == "技术领域":
            return bool(state.get("tech_domain"))
        if slot == "核心问题":
            return bool(state.get("core_problem"))
        if slot == "约束条件":
            return bool(state.get("constraints"))
        if slot == "专利号":
            return bool(state.get("constraints", {}).get("patent_id"))
        return False

    def _required_slots(self, intent: dict) -> list:
        """当前意图需要补全的槽位全集（去重，保持稳定顺序）。"""
        slots = []
        for l2_list in intent.values():
            for l2 in l2_list:
                for s in config.ENTITY_INFO.get(l2, []):
                    if s not in slots:
                        slots.append(s)
        return slots

    def _missing_slots(self, state: dict) -> list:
        """当前意图中尚未填好的槽位。"""
        return [s for s in self._required_slots(state.get("intent", {}))
                if not self._slot_filled(state, s)]

    def reset(self):
        self.state = self._default_state()
        self.intent_sm.reset()

    def get_state(self) -> dict:
        """返回可序列化的外部状态"""
        exclude = {"user_input", "history", "intent", "_post_extract_route",
                    "_response", "_request", "_trace", "_eval_result",
                    "_tool_result", "_confirm_result",
                    "_pending_slot", "_pending_intent", "_gather_query"}
        return {k: v for k, v in self.state.items() if k not in exclude}

    # ==================== 构建工作流图 ====================

    def _build_engine(self) -> WorkflowEngine:
        engine = WorkflowEngine(max_steps=20)

        # 注册节点
        engine.add_node("confirm_check",     self._node_confirm_check)
        engine.add_node("intent_recognize",  self._node_intent_recognize)
        engine.add_node("entity_extract",    self._node_entity_extract)
        engine.add_node("completeness_eval", self._node_completeness_eval)
        engine.add_node("clarify_gen",       self._node_clarify_gen)
        engine.add_node("confirm_gen",       self._node_confirm_gen)
        engine.add_node("tool_execute",      self._node_tool_execute)
        engine.add_node("response_gen",      self._node_response_gen)
        engine.add_node("feedback_handle",   self._node_feedback_handle)
        engine.add_node("chitchat_reply",    self._node_chitchat_reply)
        engine.add_node("operation_hint",    self._node_operation_hint)

        # 注册路由
        engine.add_router("confirm_check",     self._route_confirm_check)
        engine.add_router("intent_recognize",  self._route_intent_recognize)
        engine.add_router("feedback_handle",   lambda s: "entity_extract")
        engine.add_router("entity_extract",    self._route_entity_extract)
        engine.add_router("completeness_eval", self._route_completeness_eval)
        engine.add_router("tool_execute",      lambda s: "response_gen")

        # 入口
        engine.set_entry("confirm_check")
        return engine

    # ==================== 节点实现 ====================

    def _node_confirm_check(self, state: dict) -> dict:
        """检查是否处于确认等待阶段"""
        if state["phase"] != "confirming":
            state["_confirm_result"] = "not_confirming"
            return state

        text = state["user_input"].strip().lower()
        affirm = ["确认", "继续", "是的", "好的", "没问题", "对", "可以", "ok", "yes", "开始检索", "就这样"]
        # 否定/犹豫词优先：任何修改诉求都不应被当作确认而直接执行旧条件。
        deny = ["不对", "不是", "不行", "不要", "不用", "修改", "调整", "换", "重新", "重来",
                "不过", "但是", "但", "先不", "另外", "再加", "改"]

        if any(kw in text for kw in affirm) and not any(kw in text for kw in deny):
            state["_confirm_result"] = "affirm"
            state["confirmed"] = True
            state["phase"] = "executing"
        else:
            state["_confirm_result"] = "deny"
        return state

    def _node_intent_recognize(self, state: dict) -> dict:
        """意图识别: 三级 Fallback (规则→远程模型→OpenAI) + 状态机修正"""
        text = state["user_input"]
        intent = None

        # 记录触发本段信息收集的原始问题（仅首轮，后续追问回答不覆盖）
        if state.get("clarify_round", 0) == 0:
            state["_gather_query"] = text

        # ① 规则匹配（按分数取最高意图，并用置信度/分差判断是否「模糊」）
        top = self.intent_recognizer.top_intent(text)
        if top and top["score"] >= self.RULE_MIN_SCORE and top["margin"] >= self.RULE_MIN_MARGIN:
            intent = {top["level1"]: top["level2"]}
            logger.info("[规则] 意图: %s (score=%.1f, margin=%.1f)", intent, top["score"], top["margin"])
        elif top:
            logger.info("[规则] 命中但模糊 (score=%.1f, margin=%.1f), 降级", top["score"], top["margin"])

        # ② 远程微调模型
        if not intent and self.intent_model.enabled:
            logger.info("[远程模型] 规则未命中/模糊, 尝试远程模型...")
            intent = self.intent_model(text)

        # ③ OpenAI 兜底
        if not intent:
            logger.info("[LLM] 规则+远程模型未命中, OpenAI 兜底...")
            system_prompt = prompts.INTENT_RECOGNIZE_PROMPT.format(
                intent_info=json.dumps(config.INTENT_INFO, ensure_ascii=False, indent=2),
                tech_domain=state["tech_domain"] or "未知",
                core_problem=state["core_problem"] or "未知",
                constraints=json.dumps(state["constraints"], ensure_ascii=False) or "无",
            )

            result = self.llm.call_json(system_prompt, text)
            level1 = result.get("一级意图", "chitchat")
            level2 = result.get("二级意图", [])
            if level1 not in config.INTENT_INFO:
                level1 = "chitchat"
            intent = {level1: level2} if level2 else {level1: [config.INTENT_INFO[level1][0]]}
            logger.info("[LLM] 意图: %s", intent)

        # ④ 状态机修正
        intent = self.intent_sm.correct(intent, phase=state.get("phase"))

        level1 = list(intent.keys())[0]
        level2 = list(intent.values())[0]
        state["intent"] = intent
        state["intent_level1"] = level1
        state["intent_level2"] = level2
        state["phase"] = "recognized"
        return state

    def _node_entity_extract(self, state: dict) -> dict:
        """实体抽取: 规则优先, LLM 补充缺失槽位"""
        text = state["user_input"]
        intent = state["intent"]

        # 确定需要抽取的实体类型
        schema = set()
        for l1, l2_list in intent.items():
            for l2 in l2_list:
                if l2 in config.ENTITY_INFO:
                    schema.update(config.ENTITY_INFO[l2])
        if not schema:
            return state

        schema = list(schema)

        # 规则抽取
        slots = self.entity_extractor(text, schema)
        logger.info("[规则] 实体: %s", slots)

        # LLM 补充
        rest = [s for s in schema if s not in slots]
        if rest:
            logger.info("[LLM] 补充缺失: %s", rest)
            system_prompt = prompts.ENTITY_EXTRACT_PROMPT.format(
                rest=rest,
                tech_domain=state["tech_domain"] or "未知",
                core_problem=state["core_problem"] or "未知",
                constraints=json.dumps(state["constraints"], ensure_ascii=False) or "无",
            )

            llm_slots = self.llm.call_json(system_prompt, text)
            logger.info("[LLM] 实体: %s", llm_slots)
            for key in rest:
                if key in llm_slots:
                    if key == "约束条件" and isinstance(llm_slots[key], dict):
                        slots.setdefault("约束条件", {}).update(llm_slots[key])
                    else:
                        slots[key] = llm_slots[key]

        # 更新状态
        if slots.get("技术领域"):
            state["tech_domain"] = slots["技术领域"]
        if slots.get("核心问题"):
            state["core_problem"] = slots["核心问题"]
        if slots.get("约束条件") and isinstance(slots["约束条件"], dict):
            state["constraints"].update(slots["约束条件"])
        if slots.get("专利号"):
            state["constraints"]["patent_id"] = slots["专利号"]

        # 从 constraints + tech_domain 中拆分 filters (供检索服务做硬过滤)
        state["filters"] = self.search_service.build_filters(
            state["tech_domain"], state["constraints"]
        )

        # 记忆学习：若上一轮追问的槽位这一轮被补全，则记一次「问到了」
        pending = state.get("_pending_slot")
        if pending and self._slot_filled(state, pending):
            pl1, pl2 = (state.get("_pending_intent") or ("", ""))[:2] or ("", "")
            self.slot_memory.record_fill(pl1, pl2, pending)
            logger.info("[记忆] 槽位补全: %s/%s · %s", pl1, pl2, pending)
            state["_pending_slot"] = ""
            state["_pending_intent"] = ()

        return state

    def _node_completeness_eval(self, state: dict) -> dict:
        """完整性评估"""
        logger.info("[LLM] 评估完整性...")
        system_prompt = prompts.COMPLETENESS_EVAL_PROMPT.format(
            tech_domain=state["tech_domain"] or "未知",
            core_problem=state["core_problem"] or "未知",
            constraints=json.dumps(state["constraints"], ensure_ascii=False) or "无",
        )

        result = self.llm.call_json(system_prompt, "请评估当前需求的完整性。")
        state["_eval_result"] = result
        logger.info("[评估] 完整性评分: %s", result.get("completeness_score", "N/A"))
        return state

    def _node_clarify_gen(self, state: dict) -> dict:
        """生成追问（记忆层：命中模板则复用学到的话术，否则生成并沉淀）"""
        state["clarify_round"] += 1
        state["phase"] = "clarifying"
        logger.info("[追问] 第 %s/%s 轮", state["clarify_round"], state["max_clarify_round"])

        level1 = state.get("intent_level1", "")
        level2 = state["intent_level2"][0] if state.get("intent_level2") else ""

        # 确定性地算出还缺哪些槽位，并按「学到的优先级」排序
        missing = self._missing_slots(state)
        ranked = self.slot_memory.suggest_questions(level1, level2, missing)

        if ranked:
            slot, cached_q = ranked[0]
            if self.slot_memory.is_confident(level1, level2, slot):
                # 命中可信模板：直接复用，跳过 LLM 生成
                question = cached_q
                logger.info("[记忆] 复用模板提问 (%s/%s · %s)", level1, level2, slot)
            else:
                # 冷启动 / 话术尚不可信：用 LLM 生成，并把这次话术沉淀进记忆
                question = self._llm_clarify(state, slot)
            self.slot_memory.record_ask(level1, level2, slot, question)
            state["_pending_slot"] = slot
            state["_pending_intent"] = (level1, level2)
            state["_response"] = question
        else:
            # 没有可定位的具体槽位，回退到原 LLM 自由追问
            missing_desc = state.get("_eval_result", {}).get("highest_priority_missing", "更多细节")
            state["_response"] = self._llm_clarify(state, missing_desc)

        return state

    def _llm_clarify(self, state: dict, target: str) -> str:
        """用 LLM 生成一句针对 target（槽位名或缺失描述）的追问。"""
        system_prompt = prompts.CLARIFY_PROMPT.format(
            tech_domain=state["tech_domain"] or "未提及",
            core_problem=state["core_problem"] or "未提及",
            constraints=json.dumps(state["constraints"], ensure_ascii=False) or "未提及",
            target=target,
        )
        return self.llm.call_text(system_prompt, "请生成一个追问语句。")

    def _node_confirm_gen(self, state: dict) -> dict:
        """生成确认语句"""
        state["phase"] = "confirming"
        logger.info("[确认] 生成确认语句")

        system_prompt = prompts.CONFIRM_PROMPT.format(
            tech_domain=state["tech_domain"],
            core_problem=state["core_problem"],
            constraints=json.dumps(state["constraints"], ensure_ascii=False),
        )

        state["_response"] = self.llm.call_text(system_prompt, "请生成确认语句。")

        # 记忆学习：一段信息收集走通（收敛到确认）即记一次成功会话
        eval_result = state.get("_eval_result", {})
        level1 = state.get("intent_level1", "")
        level2 = state["intent_level2"][0] if state.get("intent_level2") else ""
        self.slot_memory.record_session(
            level1, level2,
            success=bool(eval_result.get("is_complete", True)),
            example=state.get("_gather_query", ""),
        )
        return state

    def _node_tool_execute(self, state: dict) -> dict:
        """工具路由与执行 (基于 milvus.json 真实数据检索)"""
        level1 = state.get("intent_level1", "search")
        level2_list = state.get("intent_level2", [])
        level2 = level2_list[0] if level2_list else "专利检索"
        logger.info("[工具] 路由到: %s/%s", level1, level2)

        if level2 in ("专利检索",) or level1 == "search":
            results = self._run_search(state)
            state["_tool_result"] = {"patents": results, "total": len(results)}
            self._attach_history(state, results, record=True)

        elif level2 == "专利详情查询":
            pid = state["constraints"].get("patent_id", "")
            patent = self.search_service.find_by_id(pid)
            if patent:
                state["_tool_result"] = patent
            else:
                state["_tool_result"] = {"message": f"未找到专利号 {pid} 的记录"}

        elif level2 in ("SWOT分析", "技术对比分析", "风险评估", "价值评估"):
            pid = state["constraints"].get("patent_id", "")
            patent = self.search_service.find_by_id(pid)
            state["_tool_result"] = {
                "analysis_type": level2,
                "patent": patent or {"patent_id": pid, "title": "未知"},
                "tech_domain": state["tech_domain"],
            }

        elif level2 == "专利聚束组合":
            results = self._run_search(state, top_k=5)
            state["_tool_result"] = {
                "bundle_name": f"{state['tech_domain'] or '技术'}综合方案",
                "patents": results,
            }
            self._attach_history(state, results, record=True)
        else:
            state["_tool_result"] = {"message": "暂不支持该操作"}

        return state

    def _node_response_gen(self, state: dict) -> dict:
        """生成结构化回复"""
        state["phase"] = "responding"
        tool_result = state.get("_tool_result", {})
        logger.info("[回复] 生成回复...")

        user_need = (
            f"技术领域: {state['tech_domain']}, "
            f"核心问题: {state['core_problem']}, "
            f"约束: {json.dumps(state['constraints'], ensure_ascii=False)}"
        )

        # 截断过长的工具结果防止 token 超限
        tool_result_str = json.dumps(tool_result, ensure_ascii=False)
        if len(tool_result_str) > 4000:
            tool_result_str = tool_result_str[:4000] + "...(已截断)"

        system_prompt = prompts.RESPONSE_GEN_PROMPT.format(
            grounding=GROUNDING_PROMPT,
            tool_result=tool_result_str,
            user_need=user_need,
        )

        state["_response"] = self.llm.call_text(system_prompt, "请生成结构化回复。")
        return state

    def _node_feedback_handle(self, state: dict) -> dict:
        """处理用户反馈"""
        logger.info("[反馈] 处理反馈...")
        text = state["user_input"]

        system_prompt = prompts.FEEDBACK_PROMPT.format(
            text=text,
            tech_domain=state["tech_domain"] or "未知",
            core_problem=state["core_problem"] or "未知",
            constraints=json.dumps(state["constraints"], ensure_ascii=False) or "无",
        )

        result = self.llm.call_json(system_prompt, text)
        corrections = result.get("corrections", {})
        for field, value in corrections.items():
            if field in ("tech_domain", "技术领域"):
                state["tech_domain"] = value
            elif field in ("core_problem", "核心问题"):
                state["core_problem"] = value
            elif isinstance(value, dict):
                state["constraints"].update(value)
            else:
                state["constraints"][field] = value
        logger.info("[反馈] 类型: %s, 说明: %s", result.get("feedback_type"), result.get("explanation"))
        return state

    def _node_chitchat_reply(self, state: dict) -> dict:
        """闲聊回复"""
        state["_response"] = "您好！我是专利技术检索助手，可以帮您查找相关专利、分析专利价值或组合专利方案。请告诉我您的技术需求吧。"
        return state

    def _node_operation_hint(self, state: dict) -> dict:
        """操作提示"""
        ops = state.get("intent_level2", [])
        hints = [f"{op}功能即将上线" for op in ops if op in ("专利收藏", "导出报告")]
        state["_request"] = "[操作提示] " + "；".join(hints) if hints else ""
        state["_response"] = "该功能正在开发中，请稍后再试。如需专利检索或分析，请告诉我您的需求。"
        return state

    # ==================== 历史记忆 / 推荐 ====================

    def _attach_history(self, state: dict, results: list, record: bool = False) -> None:
        """给工具结果附加「相似历史查询 + 历史推荐」，并沉淀本次检索。"""
        query_kwargs = dict(
            query_text=state.get("_gather_query") or state.get("user_input", ""),
            tech_domain=state.get("tech_domain", ""),
            core_problem=state.get("core_problem", ""),
            constraints=state.get("constraints", {}),
        )

        similar = self.query_history.find_similar(**query_kwargs, limit=3)
        if similar:
            state["_tool_result"]["similar_history_queries"] = [
                {
                    "query_text": item["record"].get("query_text", ""),
                    "tech_domain": item["record"].get("tech_domain", ""),
                    "similarity": item["similarity"],
                    "ts": item["record"].get("ts", ""),
                }
                for item in similar
            ]
            logger.info("[记忆] 命中 %s 条相似历史查询", len(similar))

        exclude_ids = {p.get("patent_id", "") for p in results}
        recommendations = self.query_history.recommend_patents(
            **query_kwargs, exclude_ids=exclude_ids, limit=5
        )
        if recommendations:
            state["_tool_result"]["history_recommendations"] = recommendations
            logger.info("[推荐] 基于历史检索推荐 %s 条专利", len(recommendations))

        if record:
            self.query_history.record_query(
                **query_kwargs,
                patents=results,
                session_id=state.get("session_id", ""),
            )

    # ==================== 专利检索工具 ====================

    def _run_search(self, state: dict, top_k: int = 10) -> list:
        """以当前 state 调用检索服务。"""
        return self.search_service.search(
            query=state.get("_gather_query") or state.get("user_input", ""),
            session_id=state.get("session_id", ""),
            tech_domain=state.get("tech_domain", ""),
            core_problem=state.get("core_problem", ""),
            constraints=state.get("constraints", {}),
            filters=state.get("filters", {}),
            sort_preference="relevance",
            top_k=top_k,
        )

    # ==================== 路由实现 ====================

    @staticmethod
    def _route_confirm_check(state: dict) -> str:
        r = state.get("_confirm_result", "")
        if r == "affirm":
            return "tool_execute"
        return "intent_recognize"

    @staticmethod
    def _route_intent_recognize(state: dict) -> str:
        level1 = state.get("intent_level1", "")
        level2 = state.get("intent_level2", [])

        if level1 == "chitchat":
            return "chitchat_reply"

        if level1 == "operation" and any(op in ("专利收藏", "导出报告") for op in level2):
            return "operation_hint"

        if level1 == "feedback":
            state["_post_extract_route"] = "eval"
            return "feedback_handle"

        if level1 == "analysis" or (level1 == "search" and "专利详情查询" in level2):
            state["_post_extract_route"] = "direct_execute"
            return "entity_extract"

        # search / operation(聚束) / 其他
        state["_post_extract_route"] = "eval"
        return "entity_extract"

    @staticmethod
    def _route_entity_extract(state: dict) -> str:
        if state.get("_post_extract_route") == "direct_execute":
            return "tool_execute"
        return "completeness_eval"

    @staticmethod
    def _route_completeness_eval(state: dict) -> str:
        eval_result = state.get("_eval_result", {})
        is_complete = eval_result.get("is_complete", False)
        if not is_complete and state["clarify_round"] < state["max_clarify_round"]:
            return "clarify_gen"
        return "confirm_gen"

    # ==================== 外部调用接口 ====================

    def __call__(self, text: str) -> dict:
        """处理一轮用户输入"""
        self.state["user_input"] = text
        self.state["_response"] = ""
        self.state["_request"] = ""
        self.state["history"].append({"role": "user", "content": text})

        # 执行工作流
        self.state = self.engine.run(self.state)

        # 构建响应
        response = {}
        if self.state["_request"]:
            response["request"] = self.state["_request"]
        if self.state["_response"]:
            response["message"] = self.state["_response"]
            self.state["history"].append({"role": "assistant", "content": self.state["_response"]})

        return response
