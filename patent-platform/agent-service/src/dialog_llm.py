"""大模型对话封装

无状态封装 OpenAI 兼容接口，供工作流各节点调用：
    - ``call_json`` : 单轮调用并解析 JSON 输出 (意图识别 / 实体抽取 / 评估)
    - ``call_text`` : 单轮调用返回纯文本 (追问 / 确认 / 回复生成)

对话历史由工作流的 state 统一管理，故此封装本身不保存任何会话状态，
天然支持多会话并发复用。``GROUNDING_PROMPT`` 提供可复用的"接地"规则，
供回复生成节点拼接，确保回答严格基于检索到的专利数据。
"""

import json
import logging

import config
from openai import OpenAI

logger = logging.getLogger(__name__)

# 默认采样温度：检索问答场景需稳定、可复现，故取较低值
DEFAULT_TEMPERATURE = 0.3

# 检索问答接地规则：强制基于检索数据作答、防幻觉、可溯源、有兜底
GROUNDING_PROMPT = """\
# 角色
你是一名严谨的专利技术检索助手。你的唯一信息来源是检索到的专利数据，
你不是通用问答助手，也不得依赖任何外部或自身先验知识。

# 核心准则（按优先级）
1. 忠于数据：所有结论、数字、专利号、发明人、日期、技术细节都必须能在检索结果中找到原文依据，
   严禁推测、补全或编造任何未出现的内容。
2. 可溯源：引用某条专利时必须标注其专利号（如 CN…），便于用户核对。
3. 知之为知之：若检索结果中没有用户所问的信息，明确说明"提供的专利数据中未包含该信息"，
   不要用常识或行业经验填补。
4. 缺数据兜底：当检索结果为空或与需求完全不相关时，回复：
   "未查询到相关专利信息，请补充技术领域、应用场景或关键约束条件后重试。"
5. 边界控制：对与专利检索无关的请求，礼貌说明职责范围并引导回检索主题，不展开回答。

# 输出规范
- 使用与用户一致的语言（默认简体中文），用词专业、客观、简洁。
- 命中多条专利时，先给一句总体概述，再用列表逐条呈现：
  专利号 | 标题 | 发明人 | 技术领域 | 公开日期。
- 字段缺失时标注"数据缺失"，不要留空臆造。
- 如对结果与需求的匹配度有保留，应如实指出并建议下一步细化方向。
"""


class DialogLLM:
    """OpenAI 兼容大模型封装（无状态）。"""

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=config.LLM_TIMEOUT,
        )
        self.model_name = config.LLM_MODEL_NAME

    def call_json(self, system_prompt: str, user_prompt: str) -> dict:
        """单轮调用并解析为 JSON；解析失败时回退到提取首个 JSON 片段。"""
        try:
            text = self._complete(system_prompt, user_prompt)
            return self._parse_json(text)
        except Exception as e:
            logger.error("[LLM] API调用失败: %s", e)
            return {"error": str(e)}

    def call_text(self, system_prompt: str, user_prompt: str) -> str:
        """单轮调用返回纯文本。"""
        try:
            return self._complete(system_prompt, user_prompt)
        except Exception as e:
            logger.error("[LLM] API调用失败: %s", e)
            return f"抱歉，模型服务暂时不可用（{type(e).__name__}），请稍后重试。"

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
