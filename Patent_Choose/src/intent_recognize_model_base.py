"""
远程微调模型意图识别客户端

通过 OpenAI 兼容 API 调用部署在 Linux GPU 服务器上的微调意图分类模型。
容错: 超时/网络错误/格式异常 → 返回 None, 由上层降级到 OpenAI 兜底。
"""

import json
from typing import Optional

import httpx

import config


class IntentRecognizeModelBase:
    """远程微调模型意图识别"""

    SYSTEM_PROMPT = """你是一个专利领域意图分类器。根据用户输入判断其意图。

支持的意图体系:
- search: 专利检索, 专利详情查询
- analysis: SWOT分析, 技术对比分析, 风险评估, 价值评估
- operation: 专利聚束组合, 专利收藏, 导出报告
- feedback: 结果不满意, 修改条件, 换方向
- chitchat: 闲聊, 无关输入

请严格以 JSON 格式输出:
{"一级意图": "search/analysis/operation/feedback/chitchat", "二级意图": ["具体意图"]}"""

    def __init__(self):
        self.base_url = config.REMOTE_MODEL_URL.rstrip("/") if config.REMOTE_MODEL_URL else ""
        self.model_name = config.REMOTE_MODEL_NAME
        self.api_key = config.REMOTE_MODEL_API_KEY
        self.timeout = config.REMOTE_MODEL_TIMEOUT
        self._enabled = bool(self.base_url and self.model_name)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __call__(self, text: str) -> Optional[dict]:
        """调用远程模型, 成功返回 {level1: [level2]}, 失败返回 None"""
        if not self._enabled:
            return None
        try:
            return self._request(text)
        except httpx.TimeoutException:
            print(f"    [远程模型] 超时 (>{self.timeout}s), 降级")
            return None
        except httpx.ConnectError:
            print("    [远程模型] 连接失败, 降级")
            return None
        except httpx.HTTPStatusError as e:
            print(f"    [远程模型] HTTP {e.response.status_code}, 降级")
            return None
        except Exception as e:
            print(f"    [远程模型] 异常: {e}, 降级")
            return None

    def _request(self, text: str) -> Optional[dict]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
            "max_tokens": 128,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()

        return self._parse_response(resp.json())

    def _parse_response(self, data: dict) -> Optional[dict]:
        try:
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            level1 = result.get("一级意图", "")
            level2 = result.get("二级意图", [])

            if level1 not in config.INTENT_INFO:
                print(f"    [远程模型] 非法一级意图: {level1}, 降级")
                return None

            if isinstance(level2, str):
                level2 = [level2]

            intent = {level1: level2 if level2 else [config.INTENT_INFO[level1][0]]}
            print(f"    [远程模型] 意图: {intent}")
            return intent
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"    [远程模型] 响应解析失败: {e}, 降级")
            return None
