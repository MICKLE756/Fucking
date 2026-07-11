"""测试执行工具：运行 pytest 并提取失败摘要，给 LLM 高信噪比的反馈。"""

from __future__ import annotations

import re
import subprocess

from .base import Tool, ToolResult

TIMEOUT_SECONDS = 300


class TestTool(Tool):
    name = "run_tests"
    description = "运行 pytest（可指定测试文件/表达式），返回结果摘要与失败详情。修改代码后必须用它验证。"
    args_hint = '{"target": "tests/test_x.py::test_y"（可选，默认全部）, "extra_args": "-x"（可选）}'

    def run(self, args: dict) -> ToolResult:
        cmd = ["python", "-m", "pytest", "-q", "--no-header"]
        if args.get("target"):
            cmd.append(args["target"])
        if args.get("extra_args"):
            cmd.extend(str(args["extra_args"]).split())
        try:
            proc = subprocess.run(
                cmd, cwd=self.workdir, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"测试超时（>{TIMEOUT_SECONDS}s）。请指定更小的 target。")
        output = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult(f"exit_code={proc.returncode}\n{summarize_pytest(output)}")


def summarize_pytest(output: str, max_chars: int = 12_000) -> str:
    """保留结果行 + FAILED/ERROR 行 + 最后的失败 traceback 片段。"""
    lines = output.splitlines()
    keep: list[str] = []
    for line in lines:
        if re.search(r"(FAILED|ERROR|passed|failed|error)", line):
            keep.append(line)
    tail = "\n".join(lines[-60:])
    summary = "\n".join(keep[-40:]) + "\n--- 输出末尾 ---\n" + tail
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary
