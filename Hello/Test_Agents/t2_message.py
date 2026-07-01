from hello_agents.core.message import Message
m = Message("你好", "user")
print(m.to_dict())             # {'role': 'user', 'content': '你好'}
# Message("x", "boss")        # 取消注释应当 pydantic 校验报错（role 不合法）