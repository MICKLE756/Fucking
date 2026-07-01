"""
第七章 · SearchTool 五后端最小可运行 demo
==========================================

`hello_agents.tools.builtin.search_tool.SearchTool` 支持 5 个搜索后端：
  - Tavily      （需 TAVILY_API_KEY + tavily-python）
  - SerpApi     （需 SERPAPI_API_KEY + google-search-results）
  - DuckDuckGo  （需 ddgs，免 key，但要联网）
  - SearXNG     （需自建/可达的 SearXNG 服务，免 key）
  - Perplexity  （需 PERPLEXITY_API_KEY）

它平时没被任何 Agent 默认启用——你想给 Agent 加联网搜索时，自己
`registry.register_tool(SearchTool(...))` 挂上去用。本 demo 就把这件事做出来。

为了「**离线、无需任何 API key、确定可跑**」，本 demo 只伪造各后端**最底层的网络/
客户端层**（Tavily client / SerpApi GoogleSearch / DDGS / requests.get / requests.post），
让每个后端拿到一份假的搜索结果——而 `SearchTool.run()` → `_structured_search` →
`_search_xxx` → `_format_text_response` 这些**真实代码全程被执行**（不是把整个工具 mock 掉）。

最后附一个「真·DuckDuckGo」联网调用：装了 ddgs（`pip install ddgs`）且有网就真跑，
否则自动跳过。

运行：
    python chapter07_search_tool_demo.py
"""

import os
import sys

# 让 demo 在未做 `pip install -e` 时也能直接跑：把包所在的上层目录加入 import 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hello_agents.tools.builtin.search_tool as st
from hello_agents.tools.builtin.search_tool import SearchTool, search
from hello_agents.tools.registry import ToolRegistry

# 在任何伪造之前，记下真实的 DDGS（用于结尾的真·联网调用）
_ORIG_DDGS = st.DDGS


# ---------------------------------------------------------------------------
# 各后端的「假数据」与「假客户端/假网络」——只替换最底层，真实解析代码照常跑
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeTavilyClient:
    """伪造 tavily.TavilyClient：只实现 .search()。"""

    def search(self, query, max_results=5, include_raw_content=False):
        return {
            "answer": "Python 是一门通用、易学的编程语言。",
            "results": [
                {"title": "Python 官网", "url": "https://www.python.org/",
                 "content": "官方网站，提供下载与文档。", "raw_content": "（全文）..."},
                {"title": "Python 文档", "url": "https://docs.python.org/3/",
                 "content": "标准库与语言参考。"},
            ][:max_results],
        }


class _FakeGoogleSearch:
    """伪造 serpapi.GoogleSearch：构造时收参数，get_dict() 返回假结果。"""

    def __init__(self, params):
        self.params = params

    def get_dict(self):
        return {
            "answer_box": {"answer": "Python 由 Guido van Rossum 创造。"},
            "organic_results": [
                {"title": "Python - 维基百科", "link": "https://zh.wikipedia.org/wiki/Python",
                 "snippet": "Python 是一种解释型、面向对象的高级语言。"},
                {"title": "Python 教程", "link": "https://docs.python.org/3/tutorial/",
                 "snippet": "官方入门教程。"},
            ],
        }


class _FakeDDGS:
    """伪造 ddgs.DDGS：上下文管理器 + .text()。"""

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
        {"title": "asyncio — 异步 I/O", "url": "https://docs.python.org/3/library/asyncio.html",
         "content": "asyncio 用来编写并发代码。"},
    ]
}

_FAKE_PERPLEXITY = {
    "choices": [{"message": {"content": "Python 是一门通用编程语言，广泛用于 AI 与 Web。"}}],
    "citations": ["https://www.python.org/", "https://docs.python.org/3/"],
}


def _print_result(title, payload_or_text):
    print(f"\n{'-' * 56}\n{title}\n{'-' * 56}")
    print(payload_or_text)


# ---------------------------------------------------------------------------
# 五个后端各跑一次（离线、伪造网络层，真实执行 run 全流程）
# ---------------------------------------------------------------------------
def demo_tavily():
    tool = SearchTool(backend="tavily")
    tool.tavily_client = _FakeTavilyClient()      # 伪造 client（平时由 TAVILY_API_KEY 初始化）
    if "tavily" not in tool.available_backends:
        tool.available_backends.append("tavily")
    _print_result("① Tavily（结构化）", tool.run(
        {"input": "什么是 Python", "backend": "tavily", "mode": "structured"}))


def demo_serpapi():
    st.GoogleSearch = _FakeGoogleSearch           # 伪造 serpapi.GoogleSearch
    tool = SearchTool(backend="serpapi", serpapi_key="demo-key")
    _print_result("② SerpApi（文本）", tool.run(
        {"input": "Python 是谁创造的", "backend": "serpapi"}))


def demo_duckduckgo_offline():
    st.DDGS = _FakeDDGS                            # 伪造 ddgs.DDGS
    tool = SearchTool(backend="duckduckgo")
    _print_result("③ DuckDuckGo（离线伪造，文本）", tool.run(
        {"input": "什么是 Python", "backend": "duckduckgo"}))


def demo_searxng():
    st.requests.get = lambda *a, **k: _FakeResp(_FAKE_SEARXNG)   # 伪造 HTTP GET
    tool = SearchTool(backend="searxng")
    _print_result("④ SearXNG（文本）", tool.run(
        {"input": "python asyncio", "backend": "searxng"}))


def demo_perplexity():
    st.requests.post = lambda *a, **k: _FakeResp(_FAKE_PERPLEXITY)  # 伪造 HTTP POST
    tool = SearchTool(backend="perplexity", perplexity_key="demo-key")
    _print_result("⑤ Perplexity（结构化）", tool.run(
        {"input": "什么是 Python", "backend": "perplexity", "mode": "structured"}))


def demo_via_registry():
    """工具的真实用法：注册进 ToolRegistry，再用 execute_tool 调用（Agent 走的就是这条）。"""
    st.requests.get = lambda *a, **k: _FakeResp(_FAKE_SEARXNG)
    registry = ToolRegistry()
    registry.register_tool(SearchTool(backend="searxng"))
    _print_result("⑥ 通过 ToolRegistry.execute_tool（searxng）",
                  registry.execute_tool("search", "python asyncio"))
    print("已注册工具:", registry.list_tools())


def demo_real_duckduckgo():
    print(f"\n{'=' * 56}\n真·DuckDuckGo（需 pip install ddgs 且有网，否则跳过）\n{'=' * 56}")
    st.DDGS = _ORIG_DDGS  # 还原被离线 demo 伪造的 DDGS
    if st.DDGS is None:
        print("⏭  未安装 ddgs，跳过。安装后即可真实联网：pip install ddgs")
        return
    try:
        tool = SearchTool(backend="duckduckgo")
        _print_result("真·DuckDuckGo 结果", tool.run(
            {"input": "what is an AI agent", "backend": "duckduckgo", "max_results": 3}))
    except Exception as e:  # 网络异常等
        print(f"⏭  联网搜索失败（可能无网络）：{e!r}")


def main() -> int:
    print("=" * 56)
    print("SearchTool 五后端 demo（离线伪造网络层，真实执行 run 全流程）")
    print("=" * 56)
    demo_tavily()
    demo_serpapi()
    demo_duckduckgo_offline()
    demo_searxng()
    demo_perplexity()
    demo_via_registry()
    demo_real_duckduckgo()
    print("\n✅ 五后端 demo 全部执行完毕")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
