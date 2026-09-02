"""将 scripts/ 加入 path，使 `import specdec` 在仓库根目录 pytest 下可用。"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
_scripts_str = str(_SCRIPTS)
if _scripts_str not in sys.path:
    sys.path.insert(0, _scripts_str)
