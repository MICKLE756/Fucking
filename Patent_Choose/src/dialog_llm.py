import json
import config
from openai import OpenAI


class DialogLLM:
    """使用大模型生成回复"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model_name = config.LLM_MODEL_NAME

        self.prompt_prefix = (
            "你是一个专利技术检索助手，必须严格遵守以下规则：\n"
            "1. 回答必须严格基于提供的专利信息，禁止编造任何未提及的细节\n"
            "2. 仅使用提供的专利数据回答\n"
            "3. 当专利信息不存在时，回复：'未查询到相关专利信息，请提供更多检索条件'\n"
            "4. 无论何种情况，不得使用外部知识补充回答\n"
            "5. 对于非专利相关的请求，礼貌引导用户回到专利检索主题\n"
            "专利信息："
        )
        self.messages: list[dict] = [{"role": "system", "content": ""}]

    def __call__(self, user_input: str, prompt: str = "") -> str:
        prompt = str(prompt)
        # 将 prompt 添加到 system 消息中
        self.messages[0]["content"] = self.prompt_prefix + prompt
        # 向 messages 中添加 user 消息
        self.messages.append({"role": "user", "content": user_input})
        # 发送请求
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
        )
        # 获取回复消息
        resp_message = resp.choices[0].message
        # 向 messages 中添加 assistant 消息
        self.messages.append({"role": resp_message.role, "content": resp_message.content})
        return resp_message.content

    def call_json(self, system_prompt: str, user_prompt: str) -> dict:
        """调用 LLM 并返回 JSON 输出"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            text = resp.choices[0].message.content
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # 尝试提取 JSON 片段
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
                return {"error": "无法解析模型输出", "raw": text}
        except Exception as e:
            print(f"  [LLM] API调用失败: {e}")
            return {"error": str(e)}

    def call_text(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 并返回纯文本"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  [LLM] API调用失败: {e}")
            return f"抱歉，模型服务暂时不可用（{type(e).__name__}），请稍后重试。"

    def reset(self):
        """重置对话历史"""
        self.messages = [{"role": "system", "content": ""}]
