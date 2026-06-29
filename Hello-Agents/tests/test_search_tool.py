"""SearchTool 五后端 + import-safe 离线单测。

与 examples/chapter07_search_tool_demo.py 同思路：只伪造各后端**最底层**的
网络/客户端层（Tavily client / SerpApi GoogleSearch / DDGS / requests.get /
requests.post），而 SearchTool.run() → _structured_search → _search_xxx →
_format_text_response 的真实解析/格式化代码全程被执行。整套测试离线、无需任何
API key、确定可跑。
"""

import hello_agents.tools.builtin.search_tool as st
from hello_agents.tools.builtin.search_tool import SearchTool
from hello_agents.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# 假客户端 / 假网络层
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeTavilyClient:
    def search(self, query, max_results=5, include_raw_content=False):
        return {
            "answer": "Python 是一门通用、易学的编程语言。",
            "results": [
                {"title": "Python 官网", "url": "https://www.python.org/",
                 "content": "官方网站。", "raw_content": "（全文）..."},
                {"title": "Python 文档", "url": "https://docs.python.org/3/",
                 "content": "标准库与语言参考。"},
            ][:max_results],
        }


class _FakeGoogleSearch:
    def __init__(self, params):
        self.params = params

    def get_dict(self):
        return {
            "answer_box": {"answer": "Python 由 Guido van Rossum 创造。"},
            "organic_results": [
                {"title": "Python - 维基百科", "link": "https://zh.wikipedia.org/wiki/Python",
                 "snippet": "Python 是一种解释型语言。"},
                {"title": "Python 教程", "link": "https://docs.python.org/3/tutorial/",
                 "snippet": "官方入门教程。"},
            ],
        }


class _FakeDDGS:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, query, max_results=5, backend="duckduckgo"):
        return [
            {"title": "什么是 Python", "href": "https://example.com/python",
             "body": "Python 是一门流行的编程语言。"},
            {"title": "Python 应用", "href": "https://example.com/uses",
             "body": "Web、数据科学、自动化等。"},
        ][:max_results]


_FAKE_SEARXNG = {
    "results": [
        {"title": "Python 官方文档", "url": "https://docs.python.org/3/",
         "content": "Python 是一门易学、强大的编程语言。"},
        {"title": "asyncio", "url": "https://docs.python.org/3/library/asyncio.html",
         "content": "asyncio 用来编写并发代码。"},
    ]
}

_FAKE_PERPLEXITY = {
    "choices": [{"message": {"content": "Python 是一门通用编程语言。"}}],
    "citations": ["https://www.python.org/", "https://docs.python.org/3/"],
}


# ---------------------------------------------------------------------------
# 各后端解析逻辑
# ---------------------------------------------------------------------------
def test_tavily_structured():
    tool = SearchTool(backend="tavily")
    tool.tavily_client = _FakeTavilyClient()
    if "tavily" not in tool.available_backends:
        tool.available_backends.append("tavily")

    payload = tool.run({"input": "什么是 Python", "backend": "tavily", "mode": "structured"})

    assert isinstance(payload, dict)
    assert payload["backend"] == "tavily"
    assert payload["answer"] == "Python 是一门通用、易学的编程语言。"
    assert [r["url"] for r in payload["results"]] == [
        "https://www.python.org/", "https://docs.python.org/3/"]


def test_serpapi_text(monkeypatch):
    monkeypatch.setattr(st, "GoogleSearch", _FakeGoogleSearch)
    tool = SearchTool(backend="serpapi", serpapi_key="demo-key")

    text = tool.run({"input": "Python 是谁创造的", "backend": "serpapi"})

    assert isinstance(text, str)
    assert "serpapi" in text
    assert "Guido van Rossum" in text
    assert "https://zh.wikipedia.org/wiki/Python" in text


def test_duckduckgo_offline(monkeypatch):
    monkeypatch.setattr(st, "DDGS", _FakeDDGS)
    tool = SearchTool(backend="duckduckgo")

    payload = tool.run({"input": "什么是 Python", "backend": "duckduckgo", "mode": "structured"})

    assert payload["backend"] == "duckduckgo"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["url"] == "https://example.com/python"


def test_searxng(monkeypatch):
    monkeypatch.setattr(st.requests, "get", lambda *a, **k: _FakeResp(_FAKE_SEARXNG))
    tool = SearchTool(backend="searxng")

    payload = tool.run({"input": "python asyncio", "backend": "searxng", "mode": "structured"})

    assert payload["backend"] == "searxng"
    assert [r["url"] for r in payload["results"]] == [
        "https://docs.python.org/3/",
        "https://docs.python.org/3/library/asyncio.html"]


def test_perplexity_structured(monkeypatch):
    monkeypatch.setattr(st.requests, "post", lambda *a, **k: _FakeResp(_FAKE_PERPLEXITY))
    tool = SearchTool(backend="perplexity", perplexity_key="demo-key")

    payload = tool.run({"input": "什么是 Python", "backend": "perplexity", "mode": "structured"})

    assert payload["backend"] == "perplexity"
    assert payload["answer"] == "Python 是一门通用编程语言。"
    assert [r["url"] for r in payload["results"]] == [
        "https://www.python.org/", "https://docs.python.org/3/"]


def test_via_registry(monkeypatch):
    """Agent 实际调工具的路径：register_tool → execute_tool。"""
    monkeypatch.setattr(st.requests, "get", lambda *a, **k: _FakeResp(_FAKE_SEARXNG))
    registry = ToolRegistry()
    registry.register_tool(SearchTool(backend="searxng"))

    out = registry.execute_tool("search", "python asyncio")

    assert "search" in registry.list_tools()
    assert "searxng" in out


def test_empty_query_returns_error():
    tool = SearchTool(backend="searxng")
    assert tool.run({"input": "   "}) == "错误：搜索查询不能为空"


# ---------------------------------------------------------------------------
# import-safe（A 步骤）：缺可选重依赖时核心导入不应崩
# ---------------------------------------------------------------------------
def test_core_import_safe():
    import hello_agents
    import hello_agents.tools as tools

    # 轻量工具始终可用
    assert "SearchTool" in tools.__all__
    assert "CalculatorTool" in tools.__all__
    # __all__ 中导出的名字都应真实可取到
    for name in tools.__all__:
        assert hasattr(tools, name), name
