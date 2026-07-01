"""可追踪的 ReAct Agent：在推理-行动循环的每个环节发出结构化事件。

设计要点：
- 不改动 `ReActAgent` 原有逻辑，通过子类重写 `run` 并注入 `emit` 回调，
  在「思考/行动/观察/最终答案」处发出结构化事件，供前端可视化。
- 事件为纯数据（`TraceEvent`），与传输层（SSE/WebSocket）解耦。
- 额外提供 `demo_trace()`：无需真实 LLM 即可产生一段可视化演示流程，
  用于中转/网络不稳定时仍能展示前端效果。
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Any, Optional, List, Iterator
import time

from hello_agents.agents.react_agent import ReActAgent
from hello_agents.core.message import Message


EmitFn = Callable[["TraceEvent"], None]


@dataclass
class TraceEvent:
    """Agent 执行过程中的一个结构化事件。

    type: start | thought | action | observation | final | error | end
    """
    type: str
    step: int = 0
    content: str = ""
    tool: Optional[str] = None
    tool_input: Optional[str] = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TracedReActAgent(ReActAgent):
    """在 ReAct 循环各环节发出 TraceEvent 的 Agent。

    用法：
        agent = TracedReActAgent(name="演示", llm=llm, tool_registry=reg)
        agent.run_traced("问题", emit=lambda ev: ...)
    """

    def run_traced(self, input_text: str, emit: EmitFn, **kwargs) -> str:
        """执行 ReAct 循环，同时通过 emit 回调发出结构化事件。"""
        kwargs = self._llm_kwargs(kwargs)
        self.current_history = []
        current_step = 0

        emit(TraceEvent(type="start", content=input_text))

        while current_step < self.max_steps:
            current_step += 1

            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc, question=input_text, history=history_str
            )

            messages = [{"role": "user", "content": prompt}]
            try:
                response_text = self.llm.invoke(messages, **kwargs)
            except Exception as e:  # LLM/网络错误，作为事件抛给前端而非中断进程
                emit(TraceEvent(type="error", step=current_step, content=f"LLM 调用失败: {e}"))
                emit(TraceEvent(type="end", step=current_step))
                return ""

            if not response_text:
                emit(TraceEvent(type="error", step=current_step, content="LLM 未返回有效响应"))
                break

            thought, action = self._parse_output(response_text)
            if thought:
                emit(TraceEvent(type="thought", step=current_step, content=thought))

            if not action:
                emit(TraceEvent(type="error", step=current_step, content="未能解析出有效的 Action，流程终止"))
                break

            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                emit(TraceEvent(type="final", step=current_step, content=final_answer))
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                emit(TraceEvent(type="end", step=current_step))
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                obs = "无效的 Action 格式，请检查。"
                emit(TraceEvent(type="observation", step=current_step, content=obs))
                self.current_history.append(f"Observation: {obs}")
                continue

            emit(TraceEvent(type="action", step=current_step, content=f"{tool_name}[{tool_input}]",
                            tool=tool_name, tool_input=tool_input))

            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            emit(TraceEvent(type="observation", step=current_step, content=str(observation)))

            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        emit(TraceEvent(type="final", step=current_step, content=final_answer))
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        emit(TraceEvent(type="end", step=current_step))
        return final_answer


def demo_trace(question: str = "请计算 15 * 23 + 45 等于多少？") -> Iterator[TraceEvent]:
    """产生一段脚本化的演示流程（无需真实 LLM），用于展示前端可视化效果。"""
    steps: List[TraceEvent] = [
        TraceEvent(type="start", content=question),
        TraceEvent(type="thought", step=1,
                   content="这是一个数学计算问题，我应该调用 calculate 工具来求值，而不是自己心算。"),
        TraceEvent(type="action", step=1, content="calculate[15 * 23 + 45]",
                   tool="calculate", tool_input="15 * 23 + 45"),
        TraceEvent(type="observation", step=1, content="390"),
        TraceEvent(type="thought", step=2,
                   content="工具返回 390，已经得到确定的计算结果，可以给出最终答案。"),
        TraceEvent(type="action", step=2, content="Finish[15 * 23 + 45 = 390]",
                   tool="Finish", tool_input="15 * 23 + 45 = 390"),
        TraceEvent(type="final", step=2, content="15 * 23 + 45 = 390"),
        TraceEvent(type="end", step=2),
    ]
    for ev in steps:
        yield ev
