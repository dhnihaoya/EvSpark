"""将仓库根目录加入 path，使 `import evspark` 在仓库内免安装跑 pytest 也可用。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
