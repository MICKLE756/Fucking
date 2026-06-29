"""让 `pytest`（控制台脚本，不经 `python -m`）也能 import 到本地 hello_agents 包。

CI 只 `pip install -r requirements.txt`（未做 `pip install -e .`），且用 `pytest`
直接启动——此时仓库根目录不在 sys.path 上，会 ModuleNotFoundError。这里把本目录
（hello_agents 的上层）插到 sys.path 最前，保证离线单测可直接 import。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
