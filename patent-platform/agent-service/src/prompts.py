"""工作流各节点使用的 Prompt 模板

与节点逻辑解耦，便于统一维护、review 与 A/B 调优。
所有模板均为 ``str.format`` 风格，占位符由调用方填充。
"""

# 意图识别（LLM 兜底层）
INTENT_RECOGNIZE_PROMPT = """你是意图识别专家。请分析用户输入，判断其意图。

支持的意图体系:
{intent_info}

当前会话状态:
- 技术领域: {tech_domain}
- 核心问题: {core_problem}
- 约束条件: {constraints}

请严格以 JSON 格式输出:
{{"一级意图": "search/analysis/operation/feedback/chitchat", "二级意图": ["具体意图"]}}"""

# 实体抽取（LLM 补充缺失槽位）
ENTITY_EXTRACT_PROMPT = """你是实体抽取专家。请从用户输入中提取以下信息：
{rest}

当前已知信息:
- 技术领域: {tech_domain}
- 核心问题: {core_problem}
- 约束条件: {constraints}

请严格以 JSON 格式输出，只包含从用户输入中能提取到的字段:
{{"技术领域": "xxx", "核心问题": "xxx", "约束条件": {{"key": "value"}}, "专利号": "xxx"}}
如果某个字段无法从输入中提取，请不要包含该字段。"""

# 需求完整性评估
COMPLETENESS_EVAL_PROMPT = """你是需求完整性评估专家。请评估当前已收集的信息是否已具备高质量的专利检索条件。

当前状态:
- 技术领域: {tech_domain}
- 核心问题: {core_problem}
- 约束条件: {constraints}

评估标准:
1. 技术领域是否清晰？
2. 核心问题是否明确？
3. 是否有至少一个具体约束条件？

请以 JSON 格式输出:
{{"is_complete": true/false, "completeness_score": 0.0-1.0, "missing_items": ["缺失项描述"], "highest_priority_missing": "最需要补充的信息"}}"""

# 追问澄清
CLARIFY_PROMPT = """你是追问澄清专家。请基于当前缺失的信息，生成一个简短、友好的追问语句。

规则:
- 一次只追问一个问题
- 尽量给出示例选项供用户参考
- 不要重复已经确认的信息

当前已知信息:
- 技术领域: {tech_domain}
- 核心问题: {core_problem}
- 约束条件: {constraints}

最需要补充的信息: {target}"""

# 条件确认
CONFIRM_PROMPT = """你是条件确认专家。请将当前收集到的需求归纳为检索条件，并生成一个确认语句。

格式参考:
"我理解您的需求是：寻找适用于【应用场景】、满足【性能约束】、关注【技术方向】的相关专利。接下来我将按这些条件进行检索，请确认是否继续。"

当前信息:
- 技术领域: {tech_domain}
- 核心问题: {core_problem}
- 约束条件: {constraints}"""

# 回复生成（拼接在 GROUNDING_PROMPT 之后）
RESPONSE_GEN_PROMPT = """{grounding}
# 本轮任务
你是回复生成专家。请严格基于下方工具执行结果，生成自然语言回复:
1. 先给整体总结（找到多少条相关专利）
2. 再给候选专利卡片（若有）: 专利号、标题、发明人、技术领域、公开日期
3. 若工具结果包含 history_recommendations，追加一节「📚 历史相关推荐」，逐条给出专利号、标题与推荐理由（来自历史用户的相似检索）
4. 若工具结果包含 similar_history_queries，简短提示用户曾有类似的历史查询
5. 再给推荐理由和下一步建议

工具执行结果: {tool_result}
用户原始需求: {user_need}"""

# 反馈修正
FEEDBACK_PROMPT = """你是反馈修正专家。用户对当前结果不满意，请分析反馈并给出修正建议。

用户反馈: {text}

当前状态:
- 技术领域: {tech_domain}
- 核心问题: {core_problem}
- 约束条件: {constraints}

请以 JSON 格式输出:
{{"feedback_type": "replace_field/add_constraint/narrow_scope/change_direction", "corrections": {{"field_name": "新值"}}, "explanation": "修正说明"}}"""
