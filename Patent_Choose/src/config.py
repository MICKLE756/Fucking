import os
import json
from pathlib import Path
from dotenv import load_dotenv

# ==================== 项目根目录 ====================
BASE_DIR = Path(__file__).parent.parent

load_dotenv(BASE_DIR / ".env")

# ==================== LLM 配置 (OpenAI 兜底) ====================
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# ==================== 远程微调模型配置 (意图识别, 可选) ====================
REMOTE_MODEL_URL = os.getenv("REMOTE_MODEL_URL", "")
REMOTE_MODEL_NAME = os.getenv("REMOTE_MODEL_NAME", "")
REMOTE_MODEL_API_KEY = os.getenv("REMOTE_MODEL_API_KEY", "")
REMOTE_MODEL_TIMEOUT = float(os.getenv("REMOTE_MODEL_TIMEOUT", "5"))

# ==================== 意图信息 ====================
# 一级意图 → 二级意图列表
INTENT_INFO = {
    "search": [
        "专利检索",
        "专利详情查询",
    ],
    "analysis": [
        "SWOT分析",
        "技术对比分析",
        "风险评估",
        "价值评估",
    ],
    "operation": [
        "专利聚束组合",
        "专利收藏",
        "导出报告",
    ],
    "feedback": [
        "结果不满意",
        "修改条件",
        "换方向",
    ],
    "chitchat": [
        "闲聊",
        "无关输入",
    ],
}

# 意图对应要抽取的实体
ENTITY_INFO = {
    "专利检索": ["技术领域", "核心问题", "约束条件"],
    "专利详情查询": ["专利号"],
    "SWOT分析": ["专利号", "技术领域"],
    "技术对比分析": ["专利号", "技术领域"],
    "风险评估": ["专利号", "技术领域"],
    "价值评估": ["专利号"],
    "专利聚束组合": ["技术领域", "核心问题"],
    "专利收藏": ["专利号"],
    "导出报告": ["专利号"],
    "结果不满意": [],
    "修改条件": ["约束条件"],
    "换方向": ["技术领域"],
    "闲聊": [],
    "无关输入": [],
}

# ==================== 专利数据 (从 milvus.json 加载) ====================
_PATENT_DATA_PATH = BASE_DIR / "milvus.json"


def _load_patents() -> list[dict]:
    """加载专利数据"""
    if _PATENT_DATA_PATH.exists():
        with open(_PATENT_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


PATENT_DATA: list[dict] = _load_patents()
