import pytest

from sweagent0.agent.parser import FormatError, parse_action


def test_parse_valid_action():
    text = '思考一下。\n```action\n{"tool": "bash", "args": {"command": "ls"}}\n```'
    action = parse_action(text)
    assert action.tool == "bash"
    assert action.args == {"command": "ls"}


def test_parse_json_fence():
    text = '```json\n{"tool": "submit", "args": {"summary": "done"}}\n```'
    assert parse_action(text).tool == "submit"


def test_no_block_raises():
    with pytest.raises(FormatError):
        parse_action("没有代码块")


def test_multiple_blocks_raise():
    text = '```action\n{"tool": "a"}\n```\n```action\n{"tool": "b"}\n```'
    with pytest.raises(FormatError):
        parse_action(text)


def test_invalid_json_raises():
    with pytest.raises(FormatError):
        parse_action("```action\n{oops}\n```")


def test_missing_tool_raises():
    with pytest.raises(FormatError):
        parse_action('```action\n{"args": {}}\n```')
