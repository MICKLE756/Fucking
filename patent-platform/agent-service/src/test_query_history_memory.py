"""查询历史记忆层冒烟测试（无需任何 API / 模型）

模拟多次专利检索沉淀历史，验证记忆层的三个能力：
    - 沉淀：record_query 持久化查询记录，新实例（模拟新进程）后仍可读取
    - 相似问题检索：find_similar 能召回语义相近的历史查询、过滤无关查询
    - 历史推荐：recommend_patents 聚合相似历史命中的专利并排除当前结果
"""

import tempfile
from pathlib import Path

from query_history_memory import QueryHistoryMemory


def main():
    tmp = Path(tempfile.mkdtemp()) / "query_history.json"
    mem = QueryHistoryMemory(path=tmp)

    print("=" * 60)
    print("查询历史记忆层 · 冒烟测试")
    print("=" * 60)

    print("\n[阶段1] 沉淀 3 条历史查询")
    mem.record_query(
        "我想找耐高温不粘锅涂层材料的专利",
        tech_domain="涂层材料", core_problem="耐高温",
        constraints={"time_range": "近3年"},
        patents=[{"patent_id": "CN001", "title": "耐高温陶瓷涂层", "tech_field": "涂层材料"},
                 {"patent_id": "CN002", "title": "不粘锅氟聚合物涂层", "tech_field": "涂层材料"}],
        session_id="s1",
    )
    mem.record_query(
        "查一下新能源电池散热相关专利",
        tech_domain="新能源电池", core_problem="散热",
        patents=[{"patent_id": "CN101", "title": "电池液冷散热结构", "tech_field": "新能源电池"}],
        session_id="s2",
    )
    mem.record_query(
        "高温环境下的锅具涂层技术",
        tech_domain="涂层材料", core_problem="耐高温耐磨",
        patents=[{"patent_id": "CN003", "title": "耐磨复合涂层", "tech_field": "涂层材料"},
                 {"patent_id": "CN001", "title": "耐高温陶瓷涂层", "tech_field": "涂层材料"}],
        session_id="s3",
    )
    print(f"    概览: {mem.summary()}")
    assert mem.summary()["total_records"] == 3

    print("\n[阶段2] 持久化：新建实例后历史仍在")
    mem2 = QueryHistoryMemory(path=tmp)
    assert mem2.summary()["total_records"] == 3
    print("    新实例 total_records=3 ✅")

    print("\n[阶段3] 相似问题检索")
    similar = mem2.find_similar(query_text="想找耐高温涂层的专利", tech_domain="涂层材料")
    print("    命中:", [(s["record"]["query_text"], s["similarity"]) for s in similar])
    assert similar, "应召回相似的涂层历史查询"
    assert all("电池" not in s["record"]["tech_domain"] for s in similar), "不应召回无关的电池查询"

    print("\n[阶段4] 历史推荐（排除当前已命中的 CN001）")
    recs = mem2.recommend_patents(
        query_text="想找耐高温涂层的专利", tech_domain="涂层材料",
        exclude_ids={"CN001"},
    )
    print("    推荐:", [(r["patent_id"], r["score"], r["reason"]) for r in recs])
    rec_ids = {r["patent_id"] for r in recs}
    assert "CN001" not in rec_ids, "当前结果中的专利不应重复推荐"
    assert rec_ids & {"CN002", "CN003"}, "应推荐相似历史查询命中的其他专利"

    print("\n所有断言通过 ✅")


if __name__ == "__main__":
    main()
