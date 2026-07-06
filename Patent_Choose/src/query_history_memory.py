"""查询历史记忆层（Query History Memory）

在槽位记忆（SlotMemory，记「怎么问」）之外，补一层「问过什么、查到什么」的
长期记忆，让 Agent 对用户的输入历史具备记忆与复用能力：

1. **历史沉淀**：每次检索走通后，把「原始问题 + 技术领域 / 核心问题 / 约束 +
   命中的专利」作为一条查询记录持久化下来（跨进程 / 跨会话保留）。
2. **相似问题检索**：新问题到来时，用字符 bigram Jaccard 相似度（无需向量库、
   与本项目「规则优先」的路线一致）从历史记录中检索相似的历史查询。
3. **历史推荐**：检索专利时，聚合相似历史查询命中的专利，按
   「相似度 × 时间衰减 × 命中次数」打分，推荐历史用户查到过的相关专利。

记忆持久化到 JSON 文件（默认 ``Patent_Choose/data/query_history.json``），
与 ``slot_memory.json`` 一样属于运行时状态，不纳入版本控制。
"""

import json
import math
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

# 判定「相似」的最低 Jaccard 相似度
MIN_SIMILARITY = 0.15

# 历史记录数量上限（超过按时间淘汰最旧）
MAX_RECORDS = 500

# 推荐打分的时间衰减半衰期（天）
RECENCY_HALF_LIFE_DAYS = 30.0

_TOKEN_SPLIT = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")


def _bigrams(text: str) -> set:
    """文本 → 词 token + 中文字符 bigram 集合（轻量语义指纹）。"""
    tokens = [t for t in _TOKEN_SPLIT.split(text.lower()) if t]
    grams = set(tokens)
    for tok in tokens:
        grams.update(tok[i:i + 2] for i in range(len(tok) - 1))
    return grams


def _similarity(a: str, b: str) -> float:
    """两段文本的 Jaccard 相似度（基于 bigram 集合）。"""
    ga, gb = _bigrams(a), _bigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _query_text_of(record: dict) -> str:
    """一条记录用于相似度匹配的拼接文本。"""
    constraints = record.get("constraints") or {}
    return " ".join(filter(None, (
        record.get("query_text", ""),
        record.get("tech_domain", ""),
        record.get("core_problem", ""),
        " ".join(str(v) for v in constraints.values() if isinstance(v, str)),
    )))


class QueryHistoryMemory:
    """用户查询历史的记忆 / 检索 / 推荐层（线程安全、JSON 持久化）。"""

    def __init__(self, path: Optional[Path] = None, autosave: bool = True):
        self.path = Path(path) if path else (config.BASE_DIR / "data" / "query_history.json")
        self.autosave = autosave
        self._lock = threading.RLock()
        self.records: list = []
        self.load()

    # ==================== 持久化 ====================

    def load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.records = data.get("records", [])
                except (json.JSONDecodeError, OSError):
                    self.records = []
            else:
                self.records = []

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"records": self.records}, f, ensure_ascii=False, indent=2)
            tmp.replace(self.path)

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    # ==================== 沉淀 ====================

    def record_query(
        self,
        query_text: str,
        tech_domain: str = "",
        core_problem: str = "",
        constraints: Optional[dict] = None,
        patents: Optional[list] = None,
        session_id: str = "",
    ) -> str:
        """沉淀一次已完成的检索（问题 + 条件 + 命中的专利摘要）。"""
        record = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "query_text": query_text,
            "tech_domain": tech_domain,
            "core_problem": core_problem,
            "constraints": constraints or {},
            "patents": [
                {
                    "patent_id": p.get("patent_id", ""),
                    "title": p.get("title", ""),
                    "tech_field": p.get("tech_field", ""),
                }
                for p in (patents or [])
                if p.get("patent_id")
            ],
        }
        with self._lock:
            self.records.append(record)
            if len(self.records) > MAX_RECORDS:
                self.records = self.records[-MAX_RECORDS:]
            self._maybe_save()
        return record["id"]

    # ==================== 相似问题检索 ====================

    def find_similar(
        self,
        query_text: str = "",
        tech_domain: str = "",
        core_problem: str = "",
        constraints: Optional[dict] = None,
        limit: int = 5,
        min_similarity: float = MIN_SIMILARITY,
    ) -> list:
        """检索与当前问题相似的历史查询，返回 [{record, similarity}]（相似度降序）。"""
        probe = _query_text_of({
            "query_text": query_text,
            "tech_domain": tech_domain,
            "core_problem": core_problem,
            "constraints": constraints or {},
        })
        if not probe.strip():
            return []

        with self._lock:
            scored = []
            for record in self.records:
                sim = _similarity(probe, _query_text_of(record))
                if sim >= min_similarity:
                    scored.append({"record": record, "similarity": round(sim, 3)})
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return json.loads(json.dumps(scored[:limit]))

    # ==================== 历史推荐 ====================

    def recommend_patents(
        self,
        query_text: str = "",
        tech_domain: str = "",
        core_problem: str = "",
        constraints: Optional[dict] = None,
        exclude_ids: Optional[set] = None,
        limit: int = 5,
    ) -> list:
        """基于相似历史查询，推荐历史用户查到过的相关专利。

        打分 = Σ(相似度 × 时间衰减)，同一专利被多条相似历史命中则累加，
        并附推荐理由（被多少条相似历史查询命中、最近一次是什么问题）。
        """
        exclude_ids = exclude_ids or set()
        similar = self.find_similar(
            query_text=query_text,
            tech_domain=tech_domain,
            core_problem=core_problem,
            constraints=constraints,
            limit=20,
        )
        now = datetime.now()

        candidates: dict = {}
        for item in similar:
            record, sim = item["record"], item["similarity"]
            decay = self._recency_decay(record.get("ts", ""), now)
            for patent in record.get("patents", []):
                pid = patent.get("patent_id", "")
                if not pid or pid in exclude_ids:
                    continue
                entry = candidates.setdefault(pid, {
                    "patent_id": pid,
                    "title": patent.get("title", ""),
                    "tech_field": patent.get("tech_field", ""),
                    "score": 0.0,
                    "hit_count": 0,
                    "source_queries": [],
                })
                entry["score"] += sim * decay
                entry["hit_count"] += 1
                src = record.get("query_text") or record.get("tech_domain", "")
                if src and src not in entry["source_queries"]:
                    entry["source_queries"] = (entry["source_queries"] + [src])[-3:]

        ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)[:limit]
        for entry in ranked:
            entry["score"] = round(entry["score"], 3)
            entry["reason"] = (
                f"历史上有 {entry['hit_count']} 次相似检索命中该专利"
                f"（如：{entry['source_queries'][-1]}）" if entry["source_queries"]
                else f"历史上有 {entry['hit_count']} 次相似检索命中该专利"
            )
        return ranked

    @staticmethod
    def _recency_decay(ts: str, now: datetime) -> float:
        """按记录时间做指数衰减（半衰期 RECENCY_HALF_LIFE_DAYS 天）。"""
        try:
            age_days = max((now - datetime.fromisoformat(ts)).total_seconds() / 86400.0, 0.0)
        except ValueError:
            return 1.0
        return math.exp(-math.log(2) * age_days / RECENCY_HALF_LIFE_DAYS)

    # ==================== 概览（调试 / 展示） ====================

    def summary(self) -> dict:
        """历史记忆的紧凑概览。"""
        with self._lock:
            domains: dict = {}
            for record in self.records:
                domain = record.get("tech_domain") or "(未知领域)"
                domains[domain] = domains.get(domain, 0) + 1
            return {
                "total_records": len(self.records),
                "domains": domains,
                "latest": self.records[-1]["ts"] if self.records else None,
            }
