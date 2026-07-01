from dotenv import load_dotenv
import os
from hello_agents.core.llm import HelloAgentsLLM
os.environ["NO_PROXY"] = "localhost,127.0.0.1"   # 让本地请求绕过 Clash 代理
load_dotenv()


llm = HelloAgentsLLM()
print("invoke:", llm.invoke([{"role": "user", "content": "用一句话介绍你自己"}]))
print("stream:", "".join(llm.stream_invoke([{"role": "user", "content": "如果室友是个傻逼怎么办？"}])))