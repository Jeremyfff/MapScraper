# -*- coding: utf-8 -*-
# @Author  : Yiheng Feng
# @Time    : 4/27/2025 1:42 PM
# @Function:

import ctypes
import logging
import time
from typing import Union

import win32con
import win32gui


def find_window_with_title(window_tile: Union[str, list[str]], max_retry_count=5) -> int:
    if isinstance(window_tile, str):
        window_tile = [window_tile]
    hwnd = 0
    for i in range(max_retry_count):
        fount = False
        for wt in window_tile:
            hwnd = win32gui.FindWindow(None, wt)
            if hwnd != 0:
                fount = True
        if fount:
            break
        logging.warning(f"第{i + 1}/{max_retry_count}次尝试，未找到目标窗口: {window_tile}")
        time.sleep(1)

    if hwnd == 0:
        raise Exception(f"未找到目标窗口: {window_tile}")
    return hwnd


def get_chrome_gpu_pid(hwnd) -> int:
    texts = []

    def enum_child(hwnd_child, _):
        class_name = win32gui.GetClassName(hwnd_child)
        if class_name == 'Static':
            text_length = win32gui.SendMessage(hwnd_child, win32con.WM_GETTEXTLENGTH, 0, 0)
            if text_length > 0:
                # 使用ctypes创建安全缓冲区（长度+1用于终止符）
                text_buffer = ctypes.create_unicode_buffer(text_length + 1)

                # 发送WM_GETTEXT消息填充缓冲区
                win32gui.SendMessage(
                    hwnd_child, win32con.WM_GETTEXT,
                    text_length + 1,  # wParam：缓冲区字符数（包含终止符）
                    text_buffer  # lParam：缓冲区指针
                )
                clean_text = text_buffer.value.strip('\x00')
                if clean_text:
                    texts.append(clean_text)

        return True

    win32gui.EnumChildWindows(hwnd, enum_child, None)
    if not texts:
        raise Exception("未找到文本内容")
    elif len(texts) > 1:
        logging.warning("检测到多个文本内容，仅使用第一个")
    text = texts[0]
    try:
        pid = int(text.split(":")[1].strip())
        return pid
    except Exception as e:
        raise Exception(f"[错误] 提取PID失败， {e}")


def close_chrome_gpu_popup(hwnd):
    if win32gui.IsWindowEnabled(hwnd):
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        time.sleep(0.5)  # 等待操作生效
        if not win32gui.IsWindow(hwnd):
            logging.info("弹窗已成功关闭")
            return
    raise Exception("未知错误，可能窗口已不存在")
