"""测试远程微调模型意图识别"""
import httpx
import json

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen2.5-8B-sft-all"
SYS_PROMPT = "你是一个专利领域意图分类器。根据用户输入判断其意图，以JSON格式输出。"

test_cases = [
    # 规则能命中的 (验证模型也能正确分类)
    "我想找一种汽车减振器相关的专利",
    "帮我查一下CN202520842474.4的详细信息",
    "对这个专利做一下SWOT分析",
    "这些结果不是我想要的",
    "你好",
    # 规则不易命中的 (微调模型的价值所在)
    "这个方案靠谱吗",
    "能不能缩小一点范围",
    "帮我看看有没有类似的东西",
    "这项技术的前景怎样",
    "给我推荐几个好的",
]

print(f"模型: {MODEL}")
print(f"端点: {URL}")
print("=" * 60)

for text in test_cases:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }
    try:
        resp = httpx.post(URL, json=payload, timeout=10)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        print(f"Q: {text}")
        print(f"A: {content}")
        print()
    except Exception as e:
        print(f"Q: {text}")
        print(f"ERROR: {e}")
        print()
