from dotenv import load_dotenv
import os
from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.core.llm import HelloAgentsLLM
os.environ["NO_PROXY"] = "localhost,127.0.0.1"   # 让本地请求绕过 Clash 代理
load_dotenv()


agent = SimpleAgent(name="助手", llm=HelloAgentsLLM(), system_prompt="用一句话回答")
print(agent.run("你好，介绍一下你自己"))
print(agent.run("我刚才问了你什么？"))   # 第二句能答上来 = 多轮记忆通了