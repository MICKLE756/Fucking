"""Agent基类"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config
from .lifecycle import EventType, AgentEvent, LifecycleHook

class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        on_event: LifecycleHook = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
        # 可选的异步生命周期钩子：每次发生事件时被 await 调用
        self.on_event: LifecycleHook = on_event

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent（同步）"""
        pass

    async def arun(self, input_text: str, **kwargs) -> str:
        """
        异步运行 Agent。

        默认实现：在线程池中跑同步 run，并在前后发出生命周期事件
        （AGENT_START / AGENT_FINISH / AGENT_ERROR）。
        子类若有原生异步逻辑可重写本方法。
        """
        await self.emit_event(EventType.AGENT_START, input_text=input_text)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: self.run(input_text, **kwargs)
            )
        except Exception as e:
            await self.emit_event(EventType.AGENT_ERROR, error=str(e))
            raise
        await self.emit_event(EventType.AGENT_FINISH, output=result)
        return result

    async def emit_event(self, event_type: EventType, **data):
        """若设置了 on_event 钩子，则构造事件并 await 调用它（否则静默跳过）。"""
        if self.on_event is None:
            return
        event = AgentEvent.create(event_type, self.name, **data)
        await self.on_event(event)

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        return self.__str__()
