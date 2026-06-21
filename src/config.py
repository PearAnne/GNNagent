"""运行时配置。优先读环境变量，便于本地/云端覆盖。"""

from __future__ import annotations

import os

DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "sk-eac0280f77c74dababe3dc95c9ca6e54",
)
DEFAULT_MODEL = "deepseek-chat"
