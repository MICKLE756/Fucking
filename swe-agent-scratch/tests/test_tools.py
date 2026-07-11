import pytest

from sweagent0.tools import ToolError, default_registry
from sweagent0.tools.editor import EditorTool


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a - b  # bug\n", encoding="utf-8")
    return tmp_path


def test_bash_tool(repo):
    reg = default_registry(str(repo))
    result = reg.get("bash").run({"command": "echo hello"})
    assert "hello" in result.output
    assert "exit_code=0" in result.output


def test_editor_view(repo):
    tool = EditorTool(str(repo))
    result = tool.run({"mode": "view", "path": "app.py"})
    assert "def add" in result.output
    assert "1\t" in result.output  # 带行号


def test_editor_str_replace(repo):
    tool = EditorTool(str(repo))
    tool.run({"mode": "str_replace", "path": "app.py", "old_str": "a - b", "new_str": "a + b"})
    assert "a + b" in (repo / "app.py").read_text()


def test_editor_str_replace_not_found(repo):
    tool = EditorTool(str(repo))
    with pytest.raises(ToolError):
        tool.run({"mode": "str_replace", "path": "app.py", "old_str": "不存在", "new_str": "x"})


def test_editor_str_replace_ambiguous(repo):
    (repo / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    tool = EditorTool(str(repo))
    with pytest.raises(ToolError):
        tool.run({"mode": "str_replace", "path": "dup.py", "old_str": "x = 1", "new_str": "x = 2"})


def test_editor_create(repo):
    tool = EditorTool(str(repo))
    tool.run({"mode": "create", "path": "new/mod.py", "content": "print('hi')\n"})
    assert (repo / "new" / "mod.py").exists()


def test_editor_path_escape_blocked(repo):
    tool = EditorTool(str(repo))
    with pytest.raises(ToolError):
        tool.run({"mode": "view", "path": "../../etc/passwd"})


def test_search_tool(repo):
    reg = default_registry(str(repo))
    result = reg.get("search").run({"pattern": r"def \w+", "glob": "*.py"})
    assert "app.py:1" in result.output


def test_unknown_tool(repo):
    reg = default_registry(str(repo))
    with pytest.raises(ToolError):
        reg.get("nope")
