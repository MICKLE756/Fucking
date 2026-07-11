import json

from sweagent0.agent.trajectory import ELIDED, Trajectory, build_messages


def _make_traj(n_steps: int, obs_size: int = 4000) -> Trajectory:
    traj = Trajectory(task="fix bug")
    for i in range(n_steps):
        traj.add(f"思考 {i}", "bash", {"command": "ls"}, "x" * obs_size)
    return traj


def test_build_messages_no_compression():
    traj = _make_traj(3, obs_size=100)
    msgs = build_messages("sys", "task", traj, token_budget=100_000)
    assert msgs[0]["role"] == "system"
    assert all(ELIDED not in m["content"] for m in msgs)


def test_build_messages_compresses_old_observations():
    traj = _make_traj(20, obs_size=6000)
    msgs = build_messages("sys", "task", traj, token_budget=10_000)
    contents = [m["content"] for m in msgs]
    assert any(ELIDED in c for c in contents)
    # 最近一步的观察结果必须保留
    assert ELIDED not in contents[-1]
    # assistant 思考文本始终保留
    assert any("思考 0" in c for c in contents)


def test_trajectory_save(tmp_path):
    traj = _make_traj(2, obs_size=10)
    path = tmp_path / "traj.json"
    traj.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["task"] == "fix bug"
    assert len(data["steps"]) == 2
