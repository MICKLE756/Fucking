"""文件编辑器工具：view / create / str_replace 三种模式（参考 SWE-agent 的 ACI 设计）。

str_replace 要求 old_str 在文件中唯一出现，避免误改——这是比让 LLM 输出整文件
或行号编辑更稳健的编辑接口（行号在多轮编辑后极易漂移）。
"""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolError, ToolResult

VIEW_WINDOW = 200  # view 默认显示行数


class EditorTool(Tool):
    name = "editor"
    description = (
        "文件查看与编辑。mode=view 查看文件（带行号，可指定起始行）；"
        "mode=create 新建文件；mode=str_replace 精确替换（old_str 必须在文件中唯一出现）。"
    )
    args_hint = (
        '{"mode": "view|create|str_replace", "path": "相对路径", '
        '"start_line": 1, "content": "create 用", "old_str": "...", "new_str": "..."}'
    )

    def _resolve(self, rel: str) -> Path:
        p = (Path(self.workdir) / rel).resolve()
        if not str(p).startswith(str(Path(self.workdir).resolve())):
            raise ToolError("路径越界：只能访问仓库内文件")
        return p

    def run(self, args: dict) -> ToolResult:
        self.require(args, "mode", "path")
        mode = args["mode"]
        path = self._resolve(args["path"])
        if mode == "view":
            return self._view(path, int(args.get("start_line", 1)))
        if mode == "create":
            self.require(args, "content")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
            return ToolResult(f"已创建 {args['path']}（{len(args['content'])} 字符）")
        if mode == "str_replace":
            self.require(args, "old_str", "new_str")
            return self._str_replace(path, args["old_str"], args["new_str"])
        raise ToolError(f"未知 mode: {mode}")

    def _view(self, path: Path, start_line: int) -> ToolResult:
        if not path.is_file():
            raise ToolError(f"文件不存在: {path}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        end = min(start_line - 1 + VIEW_WINDOW, len(lines))
        body = "\n".join(f"{i + 1:6d}\t{lines[i]}" for i in range(start_line - 1, end))
        suffix = "" if end >= len(lines) else f"\n...[共 {len(lines)} 行，用 start_line 继续查看]"
        return ToolResult(body + suffix)

    def _str_replace(self, path: Path, old: str, new: str) -> ToolResult:
        if not path.is_file():
            raise ToolError(f"文件不存在: {path}")
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ToolError("old_str 未在文件中找到，请先 view 确认精确内容（含缩进）")
        if count > 1:
            raise ToolError(f"old_str 出现 {count} 次，不唯一。请增加上下文使其唯一")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult("替换成功")
