"""Repo Map：给 LLM 的仓库结构概览（参考 Aider 的 tree-sitter repo map 思路）。

对 Python 文件用 ast 抽取顶层符号（类/函数签名），并按「被引用次数」给文件打分，
把有限的提示词预算优先分配给最核心的文件。非 Python 文件只列出路径。

打分 = 该文件定义的符号在其他文件中被引用（名字出现）的次数，是
PageRank 的一阶近似——足够简单，也足够有效。
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", "dist", "build"}
MAX_FILES = 400
MAX_CHARS = 12_000


def _iter_py_files(root: Path) -> list[Path]:
    files = [
        p
        for p in sorted(root.rglob("*.py"))
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
    ]
    return files[:MAX_FILES]


def _extract_symbols(path: Path) -> list[str]:
    """抽取顶层 类/函数 签名行。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    symbols = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(f"class {node.name}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    symbols.append(f"    def {item.name}({_render_args(item)})")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(f"def {node.name}({_render_args(node)})")
    return symbols


def _render_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return ", ".join(a.arg for a in node.args.args)


def _symbol_names(symbols: list[str]) -> set[str]:
    names = set()
    for s in symbols:
        stripped = s.strip()
        for prefix in ("class ", "def "):
            if stripped.startswith(prefix):
                names.add(stripped[len(prefix) :].split("(")[0])
    return names


def rank_files(root: Path) -> list[tuple[Path, list[str], int]]:
    """返回 [(文件, 符号列表, 引用得分)]，按得分降序。"""
    files = _iter_py_files(root)
    file_symbols = {p: _extract_symbols(p) for p in files}
    defined = {p: _symbol_names(syms) for p, syms in file_symbols.items()}

    scores: Counter[Path] = Counter()
    texts = {}
    for p in files:
        try:
            texts[p] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            texts[p] = ""
    for p, names in defined.items():
        for other, text in texts.items():
            if other == p:
                continue
            scores[p] += sum(text.count(name) for name in names if len(name) > 2)

    ranked = [(p, file_symbols[p], scores[p]) for p in files]
    ranked.sort(key=lambda x: -x[2])
    return ranked


def build_repo_map(workdir: str) -> str:
    root = Path(workdir)
    ranked = rank_files(root)
    if not ranked:
        return "（未发现 Python 文件）"
    parts: list[str] = []
    used = 0
    for path, symbols, score in ranked:
        rel = path.relative_to(root)
        block = f"{rel}  (引用得分 {score})\n" + "\n".join(f"  {s}" for s in symbols[:30]) + "\n"
        if used + len(block) > MAX_CHARS:
            parts.append(f"...[其余 {len(ranked) - len(parts)} 个文件省略]")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)
