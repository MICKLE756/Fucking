"""bash 工具：在工作目录下执行 shell 命令，带超时与输出截断。"""

from __future__ import annotations

import subprocess

from .base import Tool, ToolResult

TIMEOUT_SECONDS = 120
MAX_OUTPUT = 20_000


class BashTool(Tool):
    name = "bash"
    description = "在仓库根目录执行一条 bash 命令，返回 stdout+stderr。适合安装依赖、查看目录、运行脚本。"
    args_hint = '{"command": "要执行的命令"}'

    def run(self, args: dict) -> ToolResult:
        self.require(args, "command")
        try:
            proc = subprocess.run(
                args["command"],
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"命令超时（>{TIMEOUT_SECONDS}s），已终止。请换用更快的命令。")
        output = (proc.stdout or "") + (proc.stderr or "")
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + f"\n...[输出被截断，共 {len(output)} 字符]"
        return ToolResult(f"exit_code={proc.returncode}\n{output.strip()}")
