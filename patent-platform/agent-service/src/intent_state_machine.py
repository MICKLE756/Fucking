"""
意图状态机 - 基于对话上下文的意图消歧

转移规则: (当前状态, 原始一级意图) → (新状态, 修正后一级意图 或 None 表示保持)

状态定义 (5 个):
  idle             空闲
  need_gathering   正在收集检索需求
  confirming       等待用户确认条件
  result_presented 已展示检索/分析结果
  analyzing        正在进行分析任务
"""

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class IntentStateMachine:
    """意图状态机: 根据对话上下文修正原始意图识别结果"""

    # workflow phase → 状态机状态映射
    PHASE_TO_STATE = {
        "idle":        "idle",
        "clarifying":  "need_gathering",
        "confirming":  "confirming",
        "responding":  "result_presented",
        "recognized":  None,
        "executing":   None,
    }

    # 状态转移表: (当前状态, 原始一级意图) → (新状态, 修正后一级意图)
    TRANSITION_TABLE = {
        # === idle ===
        ("idle", "search"):    ("need_gathering", None),
        ("idle", "analysis"):  ("analyzing",      None),
        ("idle", "operation"): ("idle",            None),
        ("idle", "feedback"):  ("idle",            "chitchat"),
        ("idle", "chitchat"):  ("idle",            None),

        # === need_gathering ===
        ("need_gathering", "search"):    ("need_gathering", None),
        ("need_gathering", "feedback"):  ("need_gathering", "search"),
        ("need_gathering", "chitchat"):  ("need_gathering", "search"),
        ("need_gathering", "analysis"):  ("analyzing",      None),
        ("need_gathering", "operation"): ("need_gathering", None),

        # === confirming ===
        ("confirming", "search"):    ("need_gathering", None),
        ("confirming", "feedback"):  ("need_gathering", None),
        ("confirming", "analysis"):  ("analyzing",      None),
        ("confirming", "chitchat"):  ("confirming",     None),
        ("confirming", "operation"): ("confirming",     None),

        # === result_presented ===
        ("result_presented", "search"):    ("need_gathering",    None),
        ("result_presented", "feedback"):  ("need_gathering",    None),
        ("result_presented", "analysis"):  ("analyzing",         None),
        ("result_presented", "chitchat"):  ("result_presented",  None),
        ("result_presented", "operation"): ("result_presented",  None),

        # === analyzing ===
        ("analyzing", "search"):    ("need_gathering", None),
        ("analyzing", "feedback"):  ("need_gathering", None),
        ("analyzing", "analysis"):  ("analyzing",      None),
        ("analyzing", "chitchat"):  ("analyzing",      None),
        ("analyzing", "operation"): ("analyzing",      None),
    }

    def __init__(self):
        self._state = "idle"

    @property
    def state(self) -> str:
        return self._state

    def reset(self):
        self._state = "idle"

    def sync_from_phase(self, phase: str):
        """从 workflow phase 同步状态机状态"""
        mapped = self.PHASE_TO_STATE.get(phase)
        if mapped is not None:
            self._state = mapped

    def correct(self, raw_intent: dict, phase: Optional[str] = None) -> dict:
        """根据当前状态修正原始意图

        Args:
            raw_intent: {"level1": ["level2", ...]}
            phase: workflow 当前 phase

        Returns:
            修正后的意图 (同格式)
        """
        if phase is not None:
            self.sync_from_phase(phase)

        if not raw_intent:
            return raw_intent

        raw_level1 = list(raw_intent.keys())[0]

        key = (self._state, raw_level1)
        if key in self.TRANSITION_TABLE:
            new_state, corrected_level1 = self.TRANSITION_TABLE[key]
            old_state = self._state
            self._state = new_state

            if corrected_level1 is not None and corrected_level1 != raw_level1:
                # 尽量保留原始二级意图：若其在修正后一级意图下合法则沿用，
                # 否则才回退到该一级意图的默认二级意图，避免无谓地丢失信息。
                raw_level2 = raw_intent[raw_level1]
                valid_l2 = config.INTENT_INFO.get(corrected_level1, [])
                kept = [l2 for l2 in raw_level2 if l2 in valid_l2]
                corrected_level2 = kept if kept else valid_l2[:1]
                corrected = {corrected_level1: corrected_level2}
                logger.info("[状态机] %s × %s → %s | 意图修正: %s → %s",
                            old_state, raw_level1, new_state, raw_intent, corrected)
                return corrected
            else:
                if old_state != new_state:
                    logger.info("[状态机] %s → %s | 意图保持: %s", old_state, new_state, raw_intent)
                return raw_intent
        else:
            return raw_intent
