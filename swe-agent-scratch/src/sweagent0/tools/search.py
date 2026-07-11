"""搜索工具：正则搜索仓库内文本文件（纯 Python 实现，不依赖 rg/grep）。"""

from __future__ import annotations

import re
from pathlib import Path

from .base import Tool, ToolError, ToolResult

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", "dist", "build"}
MAX_MATCHES = 80


class SearchTool(Tool):
    name = "search"
    description = "在仓库内按正则搜索代码，返回 文件:行号:内容。可用 glob 限定文件（如 *.py）。"
    args_hint = '{"pattern": "正则", "glob": "*.py"（可选）}'

    def run(self, args: dict) -> ToolResult:
        self.require(args, "pattern")
        try:
            regex = re.compile(args["pattern"])
        except re.error as e:
            raise ToolError(f"正则无效: {e}") from e
        glob = args.get("glob", "*")
        matches: list[str] = []
        root = Path(self.workdir)
        for path in sorted(root.rglob(glob)):
            if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{path.relative_to(root)}:{i}:{line.strip()[:200]}")
                    if len(matches) >= MAX_MATCHES:
                        matches.append(f"...[已达 {MAX_MATCHES} 条上限，请缩小范围]")
                        return ToolResult("\n".join(matches))
        return ToolResult("\n".join(matches) if matches else "无匹配结果")
