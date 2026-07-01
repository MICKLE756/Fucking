"""搜索工具 - HelloAgents 原生搜索实现。"""

import logging
import os
from typing import Any, Dict, Iterable, List

import requests
from hello_agents.tools.base import Tool, ToolParameter
from dotenv import load_dotenv
import os

load_dotenv()

# region 可选依赖导入，缺少库不会直接崩溃，自动降级
try:
    # 将HTML网页转为简洁Markdown文本，用于网页精读
    from markdownify import markdownify
except Exception:  # 没有安装则置空，后续不再执行网页格式化
    markdownify = None  # type: ignore

try:
    # DuckDuckGo免费搜索库
    from ddgs import DDGS  # type: ignore
except Exception:
    DDGS = None  # type: ignore

try:
    # Tavily高精度联网搜索API
    from tavily import TavilyClient  # type: ignore
except Exception:
    TavilyClient = None  # type: ignore

try:
    # SerpApi谷歌搜索接口
    from serpapi import GoogleSearch  # type: ignore
except Exception:
    GoogleSearch = None  # type: ignore
# endregion

logger = logging.getLogger(__name__)

# 粗略估算：4个字符≈1个token，用于截断超长文本
CHARS_PER_TOKEN = 4
# 默认单次返回5条搜索结果
DEFAULT_MAX_RESULTS = 5
# 支持的返回格式
SUPPORTED_RETURN_MODES = {"text", "structured", "json", "dict"}
# 所有支持的搜索后端
SUPPORTED_BACKENDS = {
    "hybrid",
    "advanced",
    "tavily",     # 付费高精度网页检索
    "serpapi",    # 谷歌搜索（付费key）
    "duckduckgo", # 免费无key，公共接口
    "searxng",    # 自建私有免费搜索源
    "perplexity", # 联网大模型一站式问答（付费API）
}

def _limit_text(text: str, token_limit: int) -> str:
    """
    根据token上限截断文本，防止上下文超长。
    :param text: 原始文本
    :param token_limit: 最大token数量
    :return: 截断后的文本
    """
    char_limit = token_limit * CHARS_PER_TOKEN
    if len(text) <= char_limit:
        return text
    # 超出长度则截断并添加省略标记
    return text[:char_limit] + "... [truncated]"


def _fetch_raw_content(url: str) -> str | None:
    """
    根据URL抓取网页正文，并转为Markdown干净文本。
    :param url: 网页链接
    :return: 清理后的正文，失败返回None
    """
    try:
        # 请求网页，设置10秒超时
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        logger.debug("Failed to fetch raw content for %s: %s", url, exc)
        return None

    # 如果安装了markdownify，把HTML转为Markdown
    if markdownify is not None:
        try:
            return markdownify(response.text)  # type: ignore[arg-type]
        except Exception as exc:
            logger.debug("markdownify failed for %s: %s", url, exc)
    # 转换失败，直接返回原始网页文本
    return response.text


def _normalized_result(
    *,
    title: str,
    url: str,
    content: str,
    raw_content: str | None,
) -> Dict[str, str]:
    """
    统一规整所有搜索引擎的结果字段，抹平不同API返回结构差异。
    统一输出结构：title、url、content、raw_content（可选全文）
    """
    payload: Dict[str, str] = {
        "title": title or url,
        "url": url,
        "content": content or "",
    }
    # 网页精读全文可选追加
    if raw_content is not None:
        payload["raw_content"] = raw_content
    return payload


def _structured_payload(
    results: Iterable[Dict[str, Any]],
    *,
    backend: str,
    answer: str | None = None,
    notices: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    打包完整结构化搜索结果，统一外层结构。
    :param results: 规整后的条目列表
    :param backend: 当前使用的搜索源
    :param answer: 部分API直接给出的简短答案
    :param notices: 运行警告、降级提示
    """
    return {
        "results": list(results),
        "backend": backend,
        "answer": answer,
        "notices": list(notices or []),
    }


class SearchTool(Tool):
    """支持多后端、可返回结构化结果的搜索工具。
    1. 多API自动切换：Tavily > SerpApi > DuckDuckGo兜底
    2. 支持网页精读，抓取URL全文
    3. 支持纯文本/结构化字典两种输出格式
    4. 弱依赖，缺库自动降级，不会直接抛出异常
    """

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: str | None = None,
        serpapi_key: str | None = None,
        perplexity_key: str | None = None,
    ) -> None:
        """
        初始化搜索工具
        :param backend: 默认后端，hybrid为自动混合降级模式
        :param tavily_key: Tavily密钥，优先读取环境变量 TAVILY_API_KEY
        :param serpapi_key: SerpApi谷歌搜索密钥，优先读取环境变量
        :param perplexity_key: Perplexity联网大模型密钥
        """
        super().__init__(
            name="search",
            description=(
                "智能网页搜索引擎，支持 Tavily、SerpApi、DuckDuckGo、SearXNG、"
                "Perplexity 等后端，可返回结构化或文本化的搜索结果。"
            ),
        )
        # 初始化后端名称
        self.backend = (backend or "hybrid").lower()
        # 密钥优先级：构造函数传入 > 系统环境变量
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.perplexity_key = perplexity_key or os.getenv("PERPLEXITY_API_KEY")

        self.available_backends: list[str] = []
        self.tavily_client = None
        # 自动检测可用后端
        self._setup_backends()

    # ------------------------------------------------------------------
    # Public API 对外入口（框架规范）
    # ------------------------------------------------------------------
    def run(self, parameters: Dict[str, Any]) -> str | Dict[str, Any]:  # type: ignore[override]
        """
        工具执行入口，Agent调用此方法执行搜索。
        :param parameters: 工具入参字典
        :return: 纯文本字符串 或者 结构化字典
        """
        # 兼容两种入参名：input / query，提高框架兼容性
        query = (parameters.get("input") or parameters.get("query") or "").strip()
        if not query:
            return "错误：搜索查询不能为空"

        # 覆盖临时后端，如果不合法则切回hybrid
        backend = str(parameters.get("backend", self.backend) or "hybrid").lower()
        backend = backend if backend in SUPPORTED_BACKENDS else "hybrid"

        # 读取返回格式，非法值默认使用text文本格式
        mode = str(
            parameters.get("mode")
            or parameters.get("return_mode")
            or "text"
        ).lower()
        if mode not in SUPPORTED_RETURN_MODES:
            mode = "text"

        # 是否打开网页精读（自动抓取网页全文）
        fetch_full_page = bool(parameters.get("fetch_full_page", False))
        max_results = int(parameters.get("max_results", DEFAULT_MAX_RESULTS))
        max_tokens = int(parameters.get("max_tokens_per_source", 2000))
        loop_count = int(parameters.get("loop_count", 0))

        # 执行搜索，拿到结构化数据
        payload = self._structured_search(
            query=query,
            backend=backend,
            fetch_full_page=fetch_full_page,
            max_results=max_results,
            max_tokens=max_tokens,
            loop_count=loop_count,
        )

        # 结构化模式：直接返回字典
        if mode in {"structured", "json", "dict"}:
            return payload

        # 文本模式：把字典排版为自然语言字符串
        return self._format_text_response(query=query, payload=payload)

    def get_parameters(self) -> List[ToolParameter]:
        """
        声明工具入参，框架自动生成Function Calling的JSON Schema。
        """
        return [
            ToolParameter(
                name="input",
                type="string",
                description="搜索查询关键词",
                required=True,
            ),
        ]

    # ------------------------------------------------------------------
    # Internal helpers 内部私有方法
    # ------------------------------------------------------------------
    def _setup_backends(self) -> None:
        """
        初始化检测：密钥+依赖库双重校验，自动注册可用后端。
        后端不可用时自动降级为混合模式。
        """
        # 校验Tavily
        if self.tavily_key and TavilyClient is not None:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
                print("✅ Tavily 搜索引擎已初始化")
            except Exception as exc:
                print(f"⚠️ Tavily 初始化失败: {exc}")
        elif self.tavily_key:
            print("⚠️ 未安装 tavily-python，无法使用 Tavily 搜索")
        else:
            print("⚠️ TAVILY_API_KEY 未设置")

        # 校验SerpApi谷歌搜索
        if self.serpapi_key:
            if GoogleSearch is not None:
                self.available_backends.append("serpapi")
                print("✅ SerpApi 搜索引擎已初始化")
            else:
                print("⚠️ 未安装 google-search-results，无法使用 SerpApi 搜索")
        else:
            print("⚠️ SERPAPI_API_KEY 未设置")

        # 非法后端自动修正为hybrid
        if self.backend not in SUPPORTED_BACKENDS:
            print("⚠️ 不支持的搜索后端，将使用 hybrid 模式")
            self.backend = "hybrid"
        # 指定后端不可用时自动降级
        elif self.backend == "tavily" and "tavily" not in self.available_backends:
            print("⚠️ Tavily 不可用，将使用 hybrid 模式")
            self.backend = "hybrid"
        elif self.backend == "serpapi" and "serpapi" not in self.available_backends:
            print("⚠️ SerpApi 不可用，将使用 hybrid 模式")
            self.backend = "hybrid"

        # 混合模式提示
        if self.backend == "hybrid":
            if self.available_backends:
                print(
                    "🔧 混合搜索模式已启用，可用后端: "
                    + ", ".join(self.available_backends)
                )
            else:
                print("⚠️ 没有可用的 Tavily/SerpApi 搜索源，将回退到通用模式")

    def _structured_search(
            self,
            *,
            query: str,
            backend: str,
            fetch_full_page: bool,
            max_results: int,
            max_tokens: int,
            loop_count: int,
    ) -> Dict[str, Any]:
        """
        路由分发：根据后端名称调用对应的搜索函数。
        hybrid 等价于 advanced 自动降级策略。
        """
        target_backend = "advanced" if backend == "hybrid" else backend

        if target_backend == "tavily":
            return self._search_tavily(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
        if target_backend == "serpapi":
            return self._search_serpapi(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
        if target_backend == "duckduckgo":
            return self._search_duckduckgo(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
        if target_backend == "searxng":
            return self._search_searxng(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
        if target_backend == "perplexity":
            return self._search_perplexity(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
                loop_count=loop_count,
            )
        if target_backend == "advanced":
            return self._search_advanced(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
                loop_count=loop_count,
            )

        raise ValueError(f"Unsupported search backend: {backend}")

    def _search_tavily(
            self,
            *,
            query: str,
            fetch_full_page: bool,
            max_results: int,
            max_tokens: int,
    ) -> Dict[str, Any]:
        """调用Tavily联网搜索API，支持网页原文返回。"""
        if not self.tavily_client:
            message = "TAVILY_API_KEY 未配置或 tavily 未安装"
            raise RuntimeError(message)

        # 调用接口
        response = self.tavily_client.search(
            query=query,
            max_results=max_results,
            include_raw_content=fetch_full_page,
        )

        results = []
        for item in response.get("results", [])[:max_results]:
            raw = item.get("raw_content") if fetch_full_page else item.get("content")
            # 超长原文截断
            if raw and fetch_full_page:
                raw = _limit_text(raw, max_tokens)
            # 统一格式化字段
            results.append(
                _normalized_result(
                    title=item.get("title") or item.get("url", ""),
                    url=item.get("url", ""),
                    content=item.get("content") or "",
                    raw_content=raw,
                )
            )

        return _structured_payload(
            results,
            backend="tavily",
            answer=response.get("answer"),
        )

    def _search_serpapi(
            self,
            *,
            query: str,
            fetch_full_page: bool,
            max_results: int,
            max_tokens: int,
    ) -> Dict[str, Any]:
        """SerpApi谷歌网页搜索。"""
        if not self.serpapi_key:
            raise RuntimeError("SERPAPI_API_KEY 未配置，无法使用 SerpApi 搜索")
        if GoogleSearch is None:
            raise RuntimeError("未安装 google-search-results，无法使用 SerpApi")

        # 谷歌搜索参数，限定中文地区
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "gl": "cn",
            "hl": "zh-cn",
            "num": max_results,
        }

        response = GoogleSearch(params).get_dict()

        # 提取谷歌直接答案框内容
        answer_box = response.get("answer_box") or {}
        answer = answer_box.get("answer") or answer_box.get("snippet")

        results = []
        for item in response.get("organic_results", [])[:max_results]:
            raw_content = item.get("snippet")
            if raw_content and fetch_full_page:
                raw_content = _limit_text(raw_content, max_tokens)
            results.append(
                _normalized_result(
                    title=item.get("title") or item.get("link", ""),
                    url=item.get("link", ""),
                    content=item.get("snippet") or "",
                    raw_content=raw_content,
                )
            )

        return _structured_payload(results, backend="serpapi", answer=answer)

    def _search_duckduckgo(
            self,
            *,
            query: str,
            fetch_full_page: bool,
            max_results: int,
            max_tokens: int,
    ) -> Dict[str, Any]:
        """免费DuckDuckGo搜索，无API密钥。"""
        if DDGS is None:
            raise RuntimeError("未安装 ddgs，无法使用 DuckDuckGo 搜索")

        results: List[Dict[str, Any]] = []
        notices: List[str] = []

        try:
            with DDGS(timeout=10) as client:
                search_results = client.text(query, max_results=max_results, backend="duckduckgo")
        except Exception as exc:
            raise RuntimeError(f"DuckDuckGo 搜索失败: {exc}")

        for entry in search_results:
            url = entry.get("href") or entry.get("url")
            title = entry.get("title") or url or ""
            content = entry.get("body") or entry.get("content") or ""

            if not url or not title:
                notices.append(f"忽略不完整的 DuckDuckGo 结果: {entry}")
                continue

            raw_content = content
            # 开启精读则自动访问网页抓取全文
            if fetch_full_page and url:
                fetched = _fetch_raw_content(url)
                if fetched:
                    raw_content = _limit_text(fetched, max_tokens)

            results.append(
                _normalized_result(
                    title=title,
                    url=url,
                    content=content,
                    raw_content=raw_content,
                )
            )

        return _structured_payload(results, backend="duckduckgo", notices=notices)

    def _search_searxng(
            self,
            *,
            query: str,
            fetch_full_page: bool,
            max_results: int,
            max_tokens: int,
    ) -> Dict[str, Any]:
        """对接自建开源SearXNG搜索引擎。"""
        # 读取服务地址，默认本机
        host = os.getenv("SEARXNG_URL", "http://localhost:8888").rstrip("/")
        endpoint = f"{host}/search"

        try:
            response = requests.get(
                endpoint,
                params={
                    "q": query,
                    "format": "json",
                    "language": "zh-CN",
                    "safesearch": 1,
                    "categories": "general",
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"SearXNG 搜索失败: {exc}")

        results = []
        for entry in payload.get("results", [])[:max_results]:
            url = entry.get("url") or entry.get("link")
            title = entry.get("title") or url or ""
            if not url or not title:
                continue
            content = entry.get("content") or entry.get("snippet") or ""
            raw_content = content
            # 精读网页
            if fetch_full_page and url:
                fetched = _fetch_raw_content(url)
                if fetched:
                    raw_content = _limit_text(fetched, max_tokens)
            results.append(
                _normalized_result(
                    title=title,
                    url=url,
                    content=content,
                    raw_content=raw_content,
                )
            )

        return _structured_payload(results, backend="searxng")

    def _search_perplexity(
            self,
            *,
            query: str,
            fetch_full_page: bool,
            max_results: int,
            max_tokens: int,
            loop_count: int,
    ) -> Dict[str, Any]:
        """调用Perplexity联网大模型，直接返回带引用的答案。"""
        if not self.perplexity_key:
            raise RuntimeError("PERPLEXITY_API_KEY 未配置，无法使用 Perplexity 搜索")

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.perplexity_key}",
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources.",
                },
                {"role": "user", "content": query},
            ],
        }

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # 提取AI回答内容与引用链接
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", []) or ["https://perplexity.ai"]

        results = []
        for idx, url in enumerate(citations[:max_results], start=1):
            snippet = content if idx == 1 else "See main Perplexity response above."
            raw = _limit_text(content, max_tokens) if fetch_full_page and idx == 1 else None
            results.append(
                _normalized_result(
                    title=f"Perplexity Source {loop_count + 1}-{idx}",
                    url=url,
                    content=snippet,
                    raw_content=raw,
                )
            )

        return _structured_payload(results, backend="perplexity", answer=content)

    def _search_advanced(
            self,
            *,
            query: str,
            fetch_full_page: bool,
            max_results: int,
            max_tokens: int,
            loop_count: int,
    ) -> Dict[str, Any]:
        """
        hybrid混合降级策略：
        1. 优先Tavily；失败则切换SerpApi；最后兜底DuckDuckGo免费搜索
        """
        notices: List[str] = []
        aggregated: List[Dict[str, Any]] = []
        answer: str | None = None
        backend_used = "advanced"

        # 第一轮尝试 Tavily
        if self.tavily_client:
            try:
                tavily_payload = self._search_tavily(
                    query=query,
                    fetch_full_page=fetch_full_page,
                    max_results=max_results,
                    max_tokens=max_tokens,
                )
                if tavily_payload["results"]:
                    return tavily_payload
                notices.append("⚠️ Tavily 未返回有效结果，尝试其他搜索源")
            except Exception as exc:
                notices.append(f"⚠️ Tavily 搜索失败：{exc}")

        # 第二轮尝试 SerpApi谷歌搜索
        if self.serpapi_key and GoogleSearch is not None:
            try:
                serp_payload = self._search_serpapi(
                    query=query,
                    fetch_full_page=fetch_full_page,
                    max_results=max_results,
                    max_tokens=max_tokens,
                )
                if serp_payload["results"]:
                    serp_payload["notices"] = notices + serp_payload.get("notices", [])
                    return serp_payload
                notices.append("⚠️ SerpApi 未返回有效结果，回退到通用搜索")
            except Exception as exc:
                notices.append(f"⚠️ SerpApi 搜索失败：{exc}")

        # 最后兜底：免费DuckDuckGo
        try:
            ddg_payload = self._search_duckduckgo(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
            aggregated.extend(ddg_payload["results"])
            notices.extend(ddg_payload.get("notices", []))
            backend_used = ddg_payload.get("backend", backend_used)
        except Exception as exc:
            notices.append(f"⚠️ DuckDuckGo 搜索失败：{exc}")

        return _structured_payload(
            aggregated,
            backend=backend_used,
            answer=answer,
            notices=notices,
        )

    def _format_text_response(self, *, query: str, payload: Dict[str, Any]) -> str:
        """
        将结构化字典数据，格式化为纯文本字符串，直接喂给LLM做观察结果。
        """
        answer = payload.get("answer")
        notices = payload.get("notices") or []
        results = payload.get("results") or []
        backend = payload.get("backend", self.backend)

        lines = [f"🔍 搜索关键词：{query}", f"🧭 使用搜索源：{backend}"]
        if answer:
            lines.append(f"💡 直接答案：{answer}")

        # 逐条拼接搜索结果
        if results:
            lines.append("")
            lines.append("📚 参考来源：")
            for idx, item in enumerate(results, start=1):
                title = item.get("title") or item.get("url", "")
                lines.append(f"[{idx}] {title}")
                if item.get("content"):
                    lines.append(f"    {item['content']}")
                if item.get("url"):
                    lines.append(f"    来源: {item['url']}")
                lines.append("")
        else:
            lines.append("❌ 未找到相关搜索结果。")

        # 追加警告信息
        if notices:
            lines.append("⚠️ 注意事项：")
            for notice in notices:
                if notice:
                    lines.append(f"- {notice}")

        return "\n".join(line for line in lines if line is not None)


# -----------------------------------------------------------------------------
# 便捷全局函数，不用手动实例化类，一行调用搜索
# -----------------------------------------------------------------------------
def search(query: str, backend: str = "hybrid") -> str:
    """快捷搜索入口，默认自动混合降级"""
    tool = SearchTool(backend=backend)
    return tool.run({"input": query, "backend": backend})  # type: ignore[return-value]


def search_tavily(query: str) -> str:
    """仅使用Tavily搜索"""
    tool = SearchTool(backend="tavily")
    return tool.run({"input": query, "backend": "tavily"})  # type: ignore[return-value]


def search_serpapi(query: str) -> str:
    """仅使用SerpApi谷歌搜索"""
    tool = SearchTool(backend="serpapi")
    return tool.run({"input": query, "backend": "serpapi"})  # type: ignore[return-value]


def search_hybrid(query: str) -> str:
    """混合自动降级搜索（推荐默认调用）"""
    tool = SearchTool(backend="hybrid")
    return tool.run({"input": query, "backend": "hybrid"})  # type: ignore[return-value]


# 测试入口
if __name__ == "__main__":
    tool = SearchTool(backend="searxng")
    res = tool.run({"input": "ReAct Agent 原理", "mode": "text"})
    print(res)
