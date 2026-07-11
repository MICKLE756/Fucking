"""git 工具：检查点与回滚，让 Agent 敢于尝试并在失败后恢复。"""

from __future__ import annotations

import subprocess

from .base import Tool, ToolError, ToolResult


class GitTool(Tool):
    name = "git"
    description = (
        "git 操作。mode=diff 查看当前改动；mode=checkpoint 暂存当前状态（stash 快照，不产生提交）；"
        "mode=rollback 撤销所有未提交改动回到干净状态。"
    )
    args_hint = '{"mode": "diff|checkpoint|rollback"}'

    def _git(self, *argv: str) -> str:
        proc = subprocess.run(
            ["git", *argv], cwd=self.workdir, capture_output=True, text=True, timeout=60
        )
        return (proc.stdout or "") + (proc.stderr or "")

    def run(self, args: dict) -> ToolResult:
        self.require(args, "mode")
        mode = args["mode"]
        if mode == "diff":
            out = self._git("diff")
            return ToolResult(out[:20_000] if out.strip() else "工作区无改动")
        if mode == "checkpoint":
            out = self._git("stash", "push", "--include-untracked", "-m", "sweagent0-checkpoint")
            self._git("stash", "apply")  # 保留工作区，仅留快照
            return ToolResult(f"已创建检查点。{out.strip()}")
        if mode == "rollback":
            self._git("checkout", "--", ".")
            self._git("clean", "-fd")
            return ToolResult("已回滚所有未提交改动")
        raise ToolError(f"未知 mode: {mode}")
