from sweagent0.repomap import build_repo_map
from sweagent0.repomap.repo_map import rank_files


def _make_repo(tmp_path):
    (tmp_path / "core.py").write_text(
        "class Engine:\n    def start(self):\n        pass\n\ndef helper_func():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "user1.py").write_text("from core import Engine\ne = Engine()\n", encoding="utf-8")
    (tmp_path / "user2.py").write_text("from core import Engine\nEngine().start()\n", encoding="utf-8")
    return tmp_path


def test_rank_files_core_first(tmp_path):
    repo = _make_repo(tmp_path)
    ranked = rank_files(repo)
    assert ranked[0][0].name == "core.py"  # 被引用最多，排最前
    assert ranked[0][2] > 0


def test_build_repo_map_contains_symbols(tmp_path):
    repo = _make_repo(tmp_path)
    result = build_repo_map(str(repo))
    assert "class Engine" in result
    assert "def helper_func" in result


def test_empty_repo(tmp_path):
    assert "未发现" in build_repo_map(str(tmp_path))
