import json
from typing import Iterator, Optional, Union, TYPE_CHECKING, Any, Dict

from dotenv import load_dotenv
import os

from hello_agents.core.agent import Agent
from hello_agents.core.config import Config
from hello_agents.core.exceptions import HelloAgentsException
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.message import Message
from hello_agents.tools.builtin.calculator import CalculatorTool
from hello_agents.tools.registry import ToolRegistry

os.environ["NO_PROXY"] = "localhost,127.0.0.1"   # 让本地请求绕过 Clash 代理
load_dotenv()

def _map_parameter_type(param_type: str) -> str:
    """将自定义工具参数类型映射为OpenAI JSON Schema规范允许的基础类型

    Args:
        param_type: 工具定义里的参数字符串类型

    Returns:
        符合JSON Schema标准的类型字符串
    """
    normalized = (param_type or "").lower()
    # 匹配JSON Schema原生支持的类型
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    # 无法识别的类型统一降级为字符串
    return "string"

class FunctionCallAgent(Agent):
    """基于OpenAI原生函数调用机制的Agent"""

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            tool_registry: Optional["ToolRegistry"] = None,
            enable_tool_calling: bool = True,
            default_tool_choice: Union[str, dict] = "auto",
            max_tool_iterations: int = 3,
    ):
        """
        Args:
            name: Agent名称
            llm: 大模型实例
            system_prompt: 基础系统提示词
            config: 全局配置对象
            tool_registry: 工具注册表，存放所有可用工具
            enable_tool_calling: 是否开启函数调用能力
            default_tool_choice: 模型工具选择策略，auto/required/none或者指定工具字典
            max_tool_iterations: 最大连续工具调用轮次，防止无限循环调用工具
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations

    def _get_system_prompt(self) -> str:
        """构建系统提示词，注入工具描述"""
        base_prompt = self.system_prompt or "你是一个可靠的AI助理，能够在需要时调用工具完成任务。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        prompt = base_prompt + "\n\n## 可用工具\n"
        prompt += "当你判断需要外部信息或执行动作时，可以直接通过函数调用使用以下工具：\n"
        prompt += tools_description + "\n"
        prompt += "\n请主动决定是否调用工具，合理利用多次调用来获得完备答案。"
        return prompt

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """遍历工具注册表，把所有工具转换成OpenAI兼容的Function Calling Schema"""
        if not self.enable_tool_calling or not self.tool_registry:
            return []

        schemas: list[dict[str, Any]] = []

        # 第一部分：处理Tool对象形式注册的工具
        for tool in self.tool_registry.get_all_tools():
            properties: Dict[str, Any] = {}
            required: list[str] = []

            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            # 遍历参数，构建properties结构
            for param in parameters:
                properties[param.name] = {
                    "type": _map_parameter_type(param.type),
                    "description": param.description or ""
                }
                # 存在默认值则写入schema
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                # 必填参数加入required列表
                if getattr(param, "required", True):
                    required.append(param.name)

            # 组装单条function工具schema
            schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        # 第二部分：处理用register_function直接注册的普通函数（读取注册表内部字段）
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "输入文本"
                                }
                            },
                            "required": ["input"]
                        }
                    }
                }
            )

        return schemas

    @staticmethod
    def _extract_message_content(raw_content: Any) -> str:
        """从OpenAI响应的message.content中安全提取文本内容
        兼容字符串、多模态列表两种格式
        """
        if raw_content is None:
            return ""
        # 普通文本直接返回
        if isinstance(raw_content, str):
            return raw_content
        # 处理多模态内容（图文列表），只拼接text字段
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        # 其他类型强制转为字符串
        return str(raw_content)

    @staticmethod
    def _parse_function_call_arguments(arguments: Optional[str]) -> dict[str, Any]:
        """解析模型返回的JSON格式函数字符串参数

        Args:
            arguments: 模型输出的JSON字符串

        Returns:
            解析后的参数字典，解析失败返回空字典
        """
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # JSON格式错误，直接返回空参
            return {}

    def _convert_parameter_types(self, tool_name: str, param_dict: dict[str, Any]) -> dict[str, Any]:
        """根据工具定义的参数类型，自动转换模型输出字符串为对应数据类型
        解决模型返回字符串数字、布尔值导致工具执行报错的问题
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        # 建立参数名→类型的映射
        type_mapping = {param.name: param.type for param in tool_params}
        converted: dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            # 按照类型强制转换值
            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                # 转换失败则保留原值，不中断执行
                converted[key] = value

        return converted

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """执行被调用的工具，并把执行结果转为字符串返回

        Args:
            tool_name: 工具名称
            arguments: 原始参数字典

        Returns:
            工具运行结果文本，异常则返回错误信息
        """
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        # 优先查找Tool对象工具
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                # 自动完成参数类型转换
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                return tool.run(typed_arguments)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        # 再查找普通函数注册的工具
        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                return func(input_text)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        # 工具不存在
        return f"❌ 错误：未找到工具 '{tool_name}'"

    def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                           tool_choice: Union[str, dict], **kwargs):
        """调用底层OpenAI客户端接口，发起带function calling参数的对话请求"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("HelloAgentsLLM 未正确初始化客户端，无法执行函数调用。")

        # 填充温度、最大token等通用参数
        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        # 发起OpenAI兼容接口调用
        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

    def run(
            self,
            input_text: str,
            *,
            max_tool_iterations: Optional[int] = None,
            tool_choice: Optional[Union[str, dict]] = None,
            **kwargs,
    ) -> str:
        """
        执行函数调用范式的多轮对话主逻辑
        自动完成：模型思考→调用工具→带回结果→继续思考，直到不再调用工具

        Args:
            input_text: 用户本轮输入
            max_tool_iterations: 临时覆盖最大工具调用轮次
            tool_choice: 临时覆盖工具选择策略
            **kwargs: LLM额外采样参数

        Returns:
            模型最终自然语言回答
        """
        # 把 config 的采样参数(temperature/max_tokens)接入本次 LLM 调用
        kwargs = self._llm_kwargs(kwargs)

        messages: list[dict[str, Any]] = []
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 载入历史对话记录
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户提问
        messages.append({"role": "user", "content": input_text})

        tool_schemas = self._build_tool_schemas()
        # 没有可用工具，直接普通问答，不走函数调用流程
        if not tool_schemas:
            response_text = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response_text, "assistant"))
            return response_text

        # 轮次上限与工具策略
        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice

        current_iteration = 0
        final_response = ""

        # 多轮工具调用循环
        while current_iteration < iterations_limit:
            # 请求模型，允许调用工具
            response = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice, **kwargs,
            )

            choice = response.choices[0]
            assistant_message = choice.message
            content = self._extract_message_content(assistant_message.content)
            tool_calls = list(assistant_message.tool_calls or [])

            # 模型选择调用工具
            if tool_calls:
                # 将assistant的函数调用消息存入上下文
                assistant_payload: dict[str, Any] = {"role": "assistant", "content": content}
                assistant_payload["tool_calls"] = []

                for tool_call in tool_calls:
                    assistant_payload["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
                messages.append(assistant_payload)

                # 逐个执行每一个工具调用，并把tool结果追加到消息上下文
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                    result = self._execute_tool_call(tool_name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                        }
                    )

                # 工具轮次计数+1，进入下一轮模型思考
                current_iteration += 1
                continue

            # 模型不再调用工具，拿到最终回答，退出循环
            final_response = content
            messages.append({"role": "assistant", "content": final_response})
            break

        # 达到最大工具调用轮次仍未产出自然回答：强制禁止工具调用，让模型总结结果
        if current_iteration >= iterations_limit and not final_response:
            final_choice = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice="none",
                **kwargs,
            )
            final_response = self._extract_message_content(final_choice.choices[0].message.content)
            messages.append({"role": "assistant", "content": final_response})

        # 将本轮问答写入Agent历史记忆
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        return final_response

    def add_tool(self, tool) -> None:
        """便捷方法：向当前Agent快速注册工具
        自动初始化空注册表（如果不存在），支持MCP扩展工具自动展开
        """
        # 没有注册表就自动新建一个并开启工具调用
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 如果是MCP复合工具，自动拆分成多个子工具再注册
        if hasattr(tool, "auto_expand") and getattr(tool, "auto_expand"):
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for expanded_tool in expanded_tools:
                    self.tool_registry.register_tool(expanded_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return

        # 普通工具直接注册
        self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """按名称卸载工具，返回是否删除成功"""
        if self.tool_registry:
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False

    def list_tools(self) -> list[str]:
        """列出当前已注册的所有工具名"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """判断当前Agent是否拥有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式调用接口暂未实现，临时回退为一次性完整输出"""
        result = self.run(input_text, **kwargs)
        yield result


def _demo() -> int:
    """冒烟测试：用本地 .env（Ollama）真实跑通 FunctionCallAgent 的函数调用范式。

    注：需要你的 Ollama 模型支持 OpenAI 兼容的 tools/function-calling（如 qwen2.5、llama3.1）。
    运行方式（任选其一）：
        python hello_agents/agents/function_call_agent.py
        python -m hello_agents.agents.function_call_agent
    """
    try:
        llm = HelloAgentsLLM()
    except HelloAgentsException as e:
        print("\n⚠️  无法创建 LLM，请先在 .env 配置 LLM_MODEL_ID / LLM_BASE_URL"
              "（Ollama 可设 LLM_API_KEY=ollama）。")
        print(f"    原始错误：{e}")
        return 1

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = FunctionCallAgent(name="函数调用助手", llm=llm, tool_registry=registry)
    answer = agent.run("请调用计算器工具计算 15 * 23 + 45 等于多少？")
    print(f"\n✅ FunctionCallAgent 跑通，最终答案: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())