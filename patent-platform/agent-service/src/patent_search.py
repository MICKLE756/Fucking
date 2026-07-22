"""专利检索服务

封装基于 ``milvus.json`` 的检索逻辑，与对话工作流解耦：
    - ``build_filters`` : 由技术领域 + 约束条件推导硬过滤字段
    - ``search``        : 硬过滤 + 关键词评分，返回 Top-K 专利
    - ``find_by_id``    : 按专利号精确查找

所有方法不依赖对话状态对象，仅接收显式参数，便于单测与复用。
"""

import logging
import re
from datetime import datetime
from typing import Optional

import httpx

import config

DEFAULT_SORT_PREFERENCE = "relevance"
logger = logging.getLogger(__name__)


class PatentSearchService:
    """专利检索服务。

    配置 ``RETRIEVAL_SERVICE_URL`` 后优先调用 retrieval-service；
    未配置或远程调用失败时，回退到本地 ``milvus.json`` 关键词检索。
    """

    def __init__(self, patents: Optional[list] = None) -> None:
        self.patents = patents if patents is not None else config.PATENT_DATA
        self.retrieval_url = config.RETRIEVAL_SERVICE_URL
        self.timeout = config.RETRIEVAL_SERVICE_TIMEOUT

    # ==================== 过滤条件构建 ====================

    def build_filters(self, tech_domain: str, constraints: dict) -> dict:
        """由技术领域与约束条件推导硬过滤字段。

        产出键：``tech_field`` / ``publish_year_from`` / ``application_scene``。
        """
        filters: dict = {}

        if tech_domain:
            filters["tech_field"] = tech_domain

        year_from = self._parse_year_from(constraints.get("time_range", ""))
        if year_from is not None:
            filters["publish_year_from"] = year_from

        application = constraints.get("application", "")
        if application:
            filters["application_scene"] = application

        return filters

    @staticmethod
    def _parse_year_from(time_range: str) -> Optional[int]:
        """从时间范围文本解析检索起始年份。"""
        if not time_range:
            return None
        # "近3年" / "最近2年"
        m = re.search(r"[近最]近?\s*(\d+)\s*年", time_range)
        if m:
            return datetime.now().year - int(m.group(1))
        # "2022年以来" / "2023至今"
        m = re.search(r"(\d{4})\s*年?(?:以[来后]|至今)", time_range)
        if m:
            return int(m.group(1))
        # "2020-2024"
        m = re.search(r"(\d{4})\s*[-~到至]\s*(\d{4})", time_range)
        if m:
            return int(m.group(1))
        return None

    # ==================== 检索 ====================

    def search(
        self,
        tech_domain: str = "",
        core_problem: str = "",
        constraints: Optional[dict] = None,
        filters: Optional[dict] = None,
        query: str = "",
        session_id: str = "",
        sort_preference: str = DEFAULT_SORT_PREFERENCE,
        top_k: int = 10,
    ) -> list:
        """检索并返回最多 ``top_k`` 条专利。"""
        if self.retrieval_url:
            try:
                return self._remote_search(
                    query=query,
                    session_id=session_id,
                    tech_domain=tech_domain,
                    core_problem=core_problem,
                    constraints=constraints or {},
                    filters=filters or {},
                    sort_preference=sort_preference,
                    top_k=top_k,
                )
            except Exception:
                logger.exception("[检索服务] retrieval-service 调用失败，回退到本地 milvus.json")

        return self._local_search(
            tech_domain=tech_domain,
            core_problem=core_problem,
            constraints=constraints,
            filters=filters,
            top_k=top_k,
        )

    def _remote_search(
        self,
        query: str = "",
        session_id: str = "",
        tech_domain: str = "",
        core_problem: str = "",
        constraints: Optional[dict] = None,
        filters: Optional[dict] = None,
        sort_preference: str = DEFAULT_SORT_PREFERENCE,
        top_k: int = 10,
    ) -> list:
        """调用 retrieval-service 的 /retrieve/search。"""
        payload = {
            "query": query or " ".join(
                part for part in (tech_domain, core_problem) if part
            ),
            "session_id": session_id,
            "tech_field": tech_domain or None,
            "core_problem": core_problem or None,
            "constraints": constraints or {},
            "filters": filters or {},
            "sort_preference": sort_preference,
            "top_k": top_k,
        }
        url = f"{self.retrieval_url}/retrieve/search"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return [self._normalize_remote_item(item) for item in data.get("items", [])]

    @staticmethod
    def _normalize_remote_item(item: dict) -> dict:
        """把 retrieval-service 返回字段转成 agent-service 回复生成可用的专利字段。"""
        detail = item.get("detail") or {}
        normalized = {
            "patent_id": item.get("patent_id", ""),
            "title": item.get("title", ""),
            "inventor": item.get("inventor", ""),
            "tech_field": item.get("tech_field", ""),
            "publish_date": item.get("publish_date", ""),
            "final_score": item.get("final_score", 0.0),
            "detail": detail,
        }
        for key in ("abstract", "applicant", "application_scene", "maturity"):
            if key in detail and key not in normalized:
                normalized[key] = detail[key]
        return normalized

    def _local_search(
        self,
        tech_domain: str = "",
        core_problem: str = "",
        constraints: Optional[dict] = None,
        filters: Optional[dict] = None,
        top_k: int = 10,
    ) -> list:
        """本地硬过滤 + 关键词评分检索，返回最多 ``top_k`` 条专利。"""
        constraints = constraints or {}
        candidates = self._apply_filters(self.patents, filters or {})

        keywords = self._collect_keywords(tech_domain, core_problem, constraints)
        if not keywords:
            return candidates[:top_k]

        scored = []
        for patent in candidates:
            score = self._score(patent, keywords)
            if score > 0:
                scored.append((score, patent))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [p for _, p in scored[:top_k]]

        # 关键词无命中时退回过滤后的候选集，避免空结果
        return results or candidates[:top_k]

    def find_by_id(self, patent_id: str) -> Optional[dict]:
        """按专利号精确查找。"""
        if not patent_id:
            return None
        for patent in self.patents:
            if patent.get("patent_id") == patent_id:
                return patent
        return None

    # ==================== 内部工具 ====================

    @staticmethod
    def _apply_filters(patents: list, filters: dict) -> list:
        """按 filters 做硬过滤；过滤后为空则退回全集。"""
        if not filters:
            return patents

        year_from = filters.get("publish_year_from")
        tech_field = filters.get("tech_field", "")
        scene = filters.get("application_scene", "")
        tech_tokens = tech_field.replace("、", " ").split() if tech_field else []

        filtered = []
        for patent in patents:
            if year_from and _publish_year(patent) < year_from:
                continue
            if tech_tokens and not any(tok in patent.get("tech_field", "") for tok in tech_tokens):
                continue
            if scene and scene not in _text_pool(patent):
                continue
            filtered.append(patent)

        return filtered if filtered else patents

    @staticmethod
    def _collect_keywords(tech_domain: str, core_problem: str, constraints: dict) -> list:
        """收集用于评分的关键词（技术领域 + 核心问题 + 字符串型约束值）。"""
        keywords: list = []
        if tech_domain:
            keywords.extend(tech_domain.replace("、", " ").split())
        if core_problem:
            keywords.extend(core_problem.replace("、", " ").split())
        for value in constraints.values():
            if isinstance(value, str):
                keywords.append(value)
        return keywords

    @staticmethod
    def _score(patent: dict, keywords: list) -> int:
        """命中关键词数即为得分。"""
        pool = _text_pool(patent)
        return sum(1 for kw in keywords if kw in pool)


def _publish_year(patent: dict) -> int:
    """解析专利公开年份，失败返回 0。"""
    pub_date = patent.get("publish_date", "")
    try:
        return int(pub_date[:4]) if len(pub_date) >= 4 else 0
    except ValueError:
        return 0


def _text_pool(patent: dict) -> str:
    """拼接用于匹配的文本字段。"""
    return " ".join((
        patent.get("title", ""),
        patent.get("tech_field", ""),
        patent.get("chunk_text", ""),
    ))
