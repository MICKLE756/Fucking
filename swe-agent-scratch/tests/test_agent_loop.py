"""端到端测试：用脚本化的假 LLM 驱动 Agent 完成一次「定位→修复→测试→提交」。"""

import subprocess

from sweagent0.agent import Agent
from sweagent0.config import RunConfig


class FakeLLM:
    """按预设脚本依次返回回复。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.usage = type("U", (), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})()

    def chat(self, messages):
        self.usage.calls += 1
        return self.replies.pop(0)


def _init_repo(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    return tmp_path


def test_agent_fixes_bug_and_submits(tmp_path):
    repo = _init_repo(tmp_path)
    replies = [
        '先看代码。\n```action\n{"tool": "editor", "args": {"mode": "view", "path": "calc.py"}}\n```',
        '发现 bug，修复。\n```action\n{"tool": "editor", "args": {"mode": "str_replace", '
        '"path": "calc.py", "old_str": "a - b", "new_str": "a + b"}}\n```',
        '完成。\n```action\n{"tool": "submit", "args": {"summary": "fix add"}}\n```',
    ]
    config = RunConfig()
    config.workdir = str(repo)
    agent = Agent(config, llm=FakeLLM(replies))
    result = agent.run("add 函数返回了错误结果")

    assert result.status == "submitted"
    assert result.steps_used == 3
    assert "a + b" in result.patch  # patch 里包含修复
    assert "a + b" in (repo / "calc.py").read_text()


def test_agent_recovers_from_format_error(tmp_path):
    repo = _init_repo(tmp_path)
    replies = [
        "我忘了输出代码块",
        '补上。\n```action\n{"tool": "submit", "args": {"summary": "done"}}\n```',
    ]
    config = RunConfig()
    config.workdir = str(repo)
    agent = Agent(config, llm=FakeLLM(replies))
    result = agent.run("任务")
    assert result.status == "submitted"
    assert result.trajectory.steps[0].tool_name == "__format_error__"


def test_agent_format_error_limit(tmp_path):
    repo = _init_repo(tmp_path)
    config = RunConfig()
    config.workdir = str(repo)
    config.agent.max_consecutive_format_errors = 2
    agent = Agent(config, llm=FakeLLM(["坏回复"] * 5))
    result = agent.run("任务")
    assert result.status == "format_error_limit"
