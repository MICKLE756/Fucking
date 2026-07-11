"""submit 工具：任务完成时调用，收集最终 patch（git diff）作为提交结果。"""

from __future__ import annotations

import subprocess

from .base import Tool, ToolResult


class SubmitTool(Tool):
    name = "submit"
    description = "确认修复完成且测试通过后调用，结束任务并提交最终 patch。"
    args_hint = '{"summary": "一句话总结你做了什么修改"}'

    def run(self, args: dict) -> ToolResult:
        proc = subprocess.run(
            ["git", "diff"], cwd=self.workdir, capture_output=True, text=True, timeout=60
        )
        patch = proc.stdout or ""
        summary = args.get("summary", "")
        return ToolResult(f"SUBMIT\n{summary}\n---PATCH---\n{patch}", is_submit=True)
