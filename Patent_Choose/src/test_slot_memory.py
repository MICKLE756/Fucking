"""槽位记忆层冒烟测试（无需任何 API / 模型）

模拟多段「同类意图」的信息收集会话，验证记忆层的学习闭环：
    - 冷启动时槽位话术不可信 → 走 LLM 生成（这里用占位话术代替）
    - 多次「问→被补全」后，话术变得可信 → 后续直接复用模板（跳过 LLM）
    - 学习成果持久化到文件，新建实例（模拟新进程/新会话）后依然保留
    - suggest_questions 按「学到的优先级」排序缺失槽位
"""

import tempfile
from pathlib import Path

from intent_slot_memory import SlotMemory, default_question

L1, L2 = "search", "专利检索"
SLOTS = ["技术领域", "约束条件"]


def simulate_session(mem: SlotMemory, fill: dict, example: str):
    """模拟一段信息收集：依次追问每个缺失槽位，按 fill 决定是否补全。"""
    for slot in SLOTS:
        # 复用 or 生成：命中可信模板则直接复用，否则「LLM 生成」（占位）
        if mem.is_confident(L1, L2, slot):
            q = mem.template(L1, L2)["slots"][slot]["question"]
            used = "复用模板"
        else:
            q = default_question(slot)  # 真实代码里这步是 LLM 生成
            used = "LLM生成"
        mem.record_ask(L1, L2, slot, q)
        if fill.get(slot):
            mem.record_fill(L1, L2, slot)
        print(f"    问[{used}] {slot}: {q[:24]}…  补全={fill.get(slot, False)}")
    mem.record_session(L1, L2, success=all(fill.values()), example=example)


def main():
    tmp = Path(tempfile.mkdtemp()) / "slot_memory.json"
    mem = SlotMemory(path=tmp)

    print("=" * 60)
    print("槽位记忆层 · 学习闭环冒烟测试")
    print("=" * 60)

    print("\n[阶段1] 冷启动：前 2 段会话，话术应为『LLM生成』，槽位被补全")
    simulate_session(mem, {"技术领域": True, "约束条件": True}, "找新能源电池的专利")
    print("  --- 第 2 段 ---")
    simulate_session(mem, {"技术领域": True, "约束条件": True}, "想查半导体封装的专利")

    print("\n[断言] 两段后槽位是否已可信复用：")
    for s in SLOTS:
        conf = mem.is_confident(L1, L2, s)
        print(f"    {s}: confident={conf}")
        assert conf, f"{s} 应在 2 次成功补全后变为可复用"

    print("\n[阶段2] 第 3 段会话：应『复用模板』，不再走 LLM 生成")
    simulate_session(mem, {"技术领域": True, "约束条件": True}, "检索医疗器械专利")

    print("\n[阶段3] 持久化：新建实例（模拟新进程）后学习成果仍在")
    mem2 = SlotMemory(path=tmp)
    for s in SLOTS:
        assert mem2.is_confident(L1, L2, s), f"持久化后 {s} 应仍可复用"
    print("    新实例 is_confident 全部为 True ✅")

    print("\n[阶段4] suggest_questions 排序（缺失槽位按学习优先级）")
    ranked = mem2.suggest_questions(L1, L2, ["约束条件", "技术领域"])
    print("    排序结果:", [s for s, _ in ranked])

    print("\n[学习概览]")
    summ = mem2.summary()
    import json
    print(json.dumps(summ, ensure_ascii=False, indent=2))

    print("\n所有断言通过 ✅")


if __name__ == "__main__":
    main()
