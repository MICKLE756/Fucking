"""槽位提问记忆层（Slot-Filling Memory）

让 Agent 在「轮询追问补全信息」这件事上变得「越用越聪明」：

1. **静态模板兜底**：每个待抽取槽位都有一句默认提问话术（冷启动即可用）。
2. **记忆 / 学习**：把每次对话「问了哪些槽位、用了什么话术、是否真的被补全、整段
   是否最终走通」沉淀下来，按意图（一级/二级）聚合成可复用的提问模板。
3. **相似问题复用**：下次遇到同类意图，直接按「学到的槽位顺序 + 学到的最佳话术」
   提问，命中模板时跳过 LLM 生成，更快、更稳、更省。

「相似」目前以意图键 `(一级意图, 二级意图)` 作为代理（同意图 ≈ 同类问题），无需向量；
后续若要做到「语义相似才复用同模板」，可在 `template_key` 处接入 embedding 检索。

记忆持久化到 JSON 文件（默认 `Patent_Choose/data/slot_memory.json`），跨进程 / 跨会话
保留学习成果。该文件是运行时状态，不建议提交进 git。
"""

import json
import threading
from pathlib import Path
from typing import Optional

import config

# 槽位默认提问话术（静态兜底，冷启动可用）
DEFAULT_SLOT_QUESTIONS = {
    "技术领域": "请问您关注的是哪个技术领域呢？例如新能源电池、半导体封装、医疗器械等。",
    "核心问题": "您希望解决的核心技术问题是什么？例如提升散热效率、降低能耗、增强结构强度等。",
    "约束条件": "有没有具体的限定条件？例如时间范围、是否仅看已授权、地域、申请人等。",
    "专利号": "请提供需要查询的专利号（如 CN…/ZL…），方便我精准定位。",
}

# 复用「学到的话术」所需的最低可信度：该槽位被问过≥MIN_ASKS 次且补全率≥MIN_FILL_RATE
MIN_ASKS_FOR_REUSE = 2
MIN_FILL_RATE_FOR_REUSE = 0.5

# 每个模板最多保留多少条示例问句（仅用于人工查看 / 调试）
MAX_EXAMPLES = 10


def default_question(slot: str) -> str:
    """槽位的静态默认提问话术。"""
    return DEFAULT_SLOT_QUESTIONS.get(slot, f"请补充关于「{slot}」的信息。")


class SlotMemory:
    """槽位提问的记忆 / 学习层（线程安全、JSON 持久化）。"""

    def __init__(self, path: Optional[Path] = None, autosave: bool = True):
        self.path = Path(path) if path else (config.BASE_DIR / "data" / "slot_memory.json")
        self.autosave = autosave
        self._lock = threading.RLock()
        self.templates: dict = {}
        self.load()

    # ==================== 持久化 ====================

    def load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.templates = data.get("templates", {})
                except (json.JSONDecodeError, OSError):
                    self.templates = {}
            else:
                self.templates = {}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"templates": self.templates}, f, ensure_ascii=False, indent=2)
            tmp.replace(self.path)

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    # ==================== 内部 ====================

    @staticmethod
    def _key(level1: str, level2: str) -> str:
        return f"{level1}/{level2}" if level2 else level1

    def _template(self, level1: str, level2: str, create: bool = False) -> Optional[dict]:
        key = self._key(level1, level2)
        tpl = self.templates.get(key)
        if tpl is None and create:
            tpl = {
                "level1": level1,
                "level2": level2,
                "examples": [],
                "use_count": 0,
                "success_count": 0,
                "slots": {},
            }
            self.templates[key] = tpl
        return tpl

    @staticmethod
    def _slot_entry(tpl: dict, slot: str, create: bool = False) -> Optional[dict]:
        entry = tpl["slots"].get(slot)
        if entry is None and create:
            entry = {"question": "", "ask_count": 0, "fill_count": 0}
            tpl["slots"][slot] = entry
        return entry

    @staticmethod
    def _fill_rate(entry: dict) -> float:
        asks = entry.get("ask_count", 0)
        return entry.get("fill_count", 0) / asks if asks else 0.0

    # ==================== 查询 ====================

    def template(self, level1: str, level2: str) -> Optional[dict]:
        """只读获取某意图的模板（不存在返回 None）。"""
        with self._lock:
            tpl = self.templates.get(self._key(level1, level2))
            return json.loads(json.dumps(tpl)) if tpl else None

    def is_confident(self, level1: str, level2: str, slot: str) -> bool:
        """该槽位的学习话术是否已可信到可直接复用（跳过 LLM 生成）。"""
        with self._lock:
            tpl = self._template(level1, level2)
            if not tpl:
                return False
            entry = tpl["slots"].get(slot)
            if not entry or not entry.get("question"):
                return False
            return (entry["ask_count"] >= MIN_ASKS_FOR_REUSE
                    and self._fill_rate(entry) >= MIN_FILL_RATE_FOR_REUSE)

    def suggest_questions(self, level1: str, level2: str, missing_slots: list) -> list:
        """对缺失槽位给出「按学习优先级排序」的 (槽位, 提问话术) 列表。

        排序：学过的槽位按 ask_count 降序（最常需要的先问），未学过的按传入顺序排在后面。
        话术：可信则用学到的，否则用静态默认。
        """
        with self._lock:
            tpl = self._template(level1, level2)
            learned_order = {}
            if tpl:
                ranked = sorted(
                    tpl["slots"].items(),
                    key=lambda kv: (kv[1].get("ask_count", 0), kv[1].get("fill_count", 0)),
                    reverse=True,
                )
                learned_order = {name: i for i, (name, _) in enumerate(ranked)}

            def sort_key(slot):
                # 学过的优先（按学习排名），未学过的保持原顺序、排在后面
                return (0, learned_order[slot]) if slot in learned_order else (1, missing_slots.index(slot))

            result = []
            for slot in sorted(missing_slots, key=sort_key):
                if self.is_confident(level1, level2, slot):
                    question = tpl["slots"][slot]["question"]
                else:
                    question = default_question(slot)
                result.append((slot, question))
            return result

    # ==================== 学习 / 记录 ====================

    def record_ask(self, level1: str, level2: str, slot: str, question: str) -> None:
        """记录「为某意图的某槽位问了一句话」。"""
        with self._lock:
            tpl = self._template(level1, level2, create=True)
            entry = self._slot_entry(tpl, slot, create=True)
            entry["ask_count"] += 1
            if question:
                # 学到的话术：优先保留已有可信话术，冷启动时记录最新一条
                entry["question"] = question if not entry["question"] else entry["question"]
            self._maybe_save()

    def record_fill(self, level1: str, level2: str, slot: str) -> None:
        """记录「之前追问的某槽位这一轮被成功补全」。"""
        with self._lock:
            tpl = self._template(level1, level2, create=True)
            entry = self._slot_entry(tpl, slot, create=True)
            entry["fill_count"] += 1
            self._maybe_save()

    def record_session(self, level1: str, level2: str, success: bool,
                       example: str = "") -> None:
        """记录「一段信息收集会话结束」及其是否最终走通。"""
        with self._lock:
            tpl = self._template(level1, level2, create=True)
            tpl["use_count"] += 1
            if success:
                tpl["success_count"] += 1
            if example and example not in tpl["examples"]:
                tpl["examples"].append(example)
                tpl["examples"] = tpl["examples"][-MAX_EXAMPLES:]
            self._maybe_save()

    # ==================== 概览（调试 / 展示） ====================

    def summary(self) -> dict:
        """返回各意图模板的紧凑概览，便于人工查看学习成果。"""
        with self._lock:
            out = {}
            for key, tpl in self.templates.items():
                out[key] = {
                    "use_count": tpl["use_count"],
                    "success_rate": round(
                        tpl["success_count"] / tpl["use_count"], 3) if tpl["use_count"] else 0.0,
                    "slots": {
                        s: {
                            "ask": e["ask_count"],
                            "fill": e["fill_count"],
                            "fill_rate": round(self._fill_rate(e), 3),
                            "confident": self.is_confident(tpl["level1"], tpl["level2"], s),
                            "question": e["question"] or default_question(s),
                        }
                        for s, e in tpl["slots"].items()
                    },
                }
            return out
