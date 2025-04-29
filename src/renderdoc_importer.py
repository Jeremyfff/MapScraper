# -*- coding: utf-8 -*-
# @Author  : Yiheng Feng
# @Time    : 4/29/2025 9:21 PM
# @Function:
# 构建 win64 目录路径
import os
import sys
import logging
win64_path = os.path.join(".", "bin", "win64")

# 添加路径到 sys.path
if win64_path not in sys.path:
    sys.path.insert(0, win64_path)
try:
    import renderdoc
    logging.info("renderdoc 模块已加载")
except Exception as e:
    raise Exception(f"renderdoc 模块导入失败, {e}")

rd = renderdoc