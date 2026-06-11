"""大模型对话封装

统一封装 OpenAI 兼容接口，供工作流各节点调用：
    - ``__call__``     : 带多轮历史的检索问答 (强约束、防幻觉)
    - ``call_json``    : 单轮调用并解析 JSON 输出 (意图识别 / 实体抽取 / 评估)
    - ``call_text``    : 单轮调用返回纯文本 (追问 / 确认 / 回复生成)
"""

import json
import config
from openai import OpenAI

# 默认采样温度：检索问答场景需稳定、可复现，故取较低值
DEFAULT_TEMPERATURE = 0.3

# 检索问答系统提示词
#   - 强制基于检索到的专利数据作答，杜绝幻觉
#   - 要求引用专利号、保持结构化、对缺数据/越界请求有明确兜底策略
SYSTEM_PROMPT = """\
# 角色
你是一名严谨的专利技术检索助手。你的唯一信息来源是【专利信息】区块中检索到的专利数据，
你不是通用问答助手，也不得依赖任何外部或自身先验知识。

# 核心准则（按优先级）
1. 忠于数据：所有结论、数字、专利号、发明人、日期、技术细节都必须能在【专利信息】中找到原文依据，
   严禁推测、补全或编造任何未出现的内容。
2. 可溯源：引用某条专利时必须标注其专利号（如 CN…），便于用户核对。
3. 知之为知之：若【专利信息】中没有用户所问的信息，明确说明"提供的专利数据中未包含该信息"，
   不要用常识或行业经验填补。
4. 缺数据兜底：当【专利信息】为空或与需求完全不相关时，回复：
   "未查询到相关专利信息，请补充技术领域、应用场景或关键约束条件后重试。"
5. 边界控制：对与专利检索无关的请求（闲聊、其他领域问答等），礼貌说明你的职责范围，
   并引导用户回到专利检索主题，不展开回答。

# 输出规范
- 使用与用户一致的语言（默认简体中文），用词专业、客观、简洁。
- 命中多条专利时，先给一句总体概述（共找到多少条、覆盖哪些方向），
  再用列表逐条呈现：专利号 | 标题 | 发明人 | 技术领域 | 公开日期。
- 命中信息不全的字段，标注"数据缺失"，不要留空臆造。
- 如对检索结果与用户需求的匹配度有保留，应如实指出并建议下一步细化方向。

# 专利信息
"""


class DialogLLM:
    """OpenAI 兼容大模型封装。"""

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model_name = config.LLM_MODEL_NAME
        self.prompt_prefix = SYSTEM_PROMPT
        self.messages: list[dict] = [{"role": "system", "content": ""}]

    def __call__(self, user_input: str, prompt: str = "") -> str:
        """带历史的检索问答：``prompt`` 为检索到的专利信息上下文。"""
        # 每轮以最新的专利信息刷新 system 提示，保证回答锚定当前检索结果
        self.messages[0]["content"] = self.prompt_prefix + str(prompt)
        self.messages.append({"role": "user", "content": user_input})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=DEFAULT_TEMPERATURE,
        )
        message = resp.choices[0].message
        self.messages.append({"role": message.role, "content": message.content})
        return message.content

    def call_json(self, system_prompt: str, user_prompt: str) -> dict:
        """单轮调用并解析为 JSON；解析失败时回退到提取首个 JSON 片段。"""
        try:
            text = self._complete(system_prompt, user_prompt)
            return self._parse_json(text)
        except Exception as e:
            print(f"  [LLM] API调用失败: {e}")
            return {"error": str(e)}

    def call_text(self, system_prompt: str, user_prompt: str) -> str:
        """单轮调用返回纯文本。"""
        try:
            return self._complete(system_prompt, user_prompt)
        except Exception as e:
            print(f"  [LLM] API调用失败: {e}")
            return f"抱歉，模型服务暂时不可用（{type(e).__name__}），请稍后重试。"

    def reset(self) -> None:
        """重置多轮对话历史。"""
        self.messages = [{"role": "system", "content": ""}]

    # ==================== 内部工具 ====================

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """发起一次无状态(单轮)补全，返回文本内容。"""
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=DEFAULT_TEMPERATURE,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _parse_json(text: str) -> dict:
        """解析模型输出为 dict，容忍 ```json 包裹或前后多余文本。"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"error": "无法解析模型输出", "raw": text}
