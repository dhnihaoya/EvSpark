"""将仓库根目录与 scripts/ 加入 path，使 `import evspark` 与 `import regulatory_design` 在仓库内免安装跑 pytest 也可用。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
