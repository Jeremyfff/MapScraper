# -*- coding: utf-8 -*-
# @Author  : Yiheng Feng
# @Time    : 4/27/2025 2:15 PM
# @Function:
import logging
import os
import sys
import threading
import time
from typing import Callable

# 构建 win64 目录路径
win64_path = os.path.join(".", "bin", "win64")

# 添加路径到 sys.path
if win64_path not in sys.path:
    sys.path.insert(0, win64_path)

try:
    import renderdoc as rd

    logging.info("renderdoc 模块已加载")
except Exception as e:
    logging.error(f"renderdoc 导入失败: {e}")
    exit(1)

