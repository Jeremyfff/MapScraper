# -*- coding: utf-8 -*-
# @Author  : Yiheng Feng
# @Time    : 4/27/2025 2:00 PM
# @Function:
import atexit
import logging
import os
import random
import re
import time
from io import BytesIO
from typing import Optional, Dict

import cv2
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

driver: Optional[webdriver.Chrome] = None


def open_driver():
    global driver
    if driver is not None:
        logging.info("driver already exist, use exist one.")
        return driver
    os.environ['RENDERDOC_HOOK_EGL'] = "0"
    service = Service()
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu-sandbox")
    options.add_argument("--gpu-startup-dialog")

    # 启动浏览器
    driver = webdriver.Chrome(service=service, options=options)

    atexit.register(close_driver)
    return driver


def close_driver():
    global driver
    if driver is not None:
        driver.quit()
        driver = None
        logging.info("chrome driver closed")
    else:
        logging.warning("driver is None, can not close")



def perform_viewport_drag():
    """
    基于视口坐标的水平拖拽（防止越界）

    参数:
        driver: WebDriver实例
        start_x_ratio: 起始X坐标比例 (0.0~1.0)
        end_x_ratio: 结束X坐标比例 (0.0~1.0)
        y_ratio: Y轴位置比例 (0.0~1.0)
        duration: 滑动持续时间（秒）
    """
    assert driver is not None, "driver is None, please open it first"
    # 获取视口尺寸和滚动位置
    try:
        viewport_width = driver.execute_script("return window.innerWidth")
        viewport_height = driver.execute_script("return window.innerHeight")

        # 计算绝对坐标（从左侧5px到右侧5px，Y轴中间位置）
        start_x = 5
        end_x = (viewport_width - 5)
        y_pos = (viewport_height // 2)

        # 创建动作链
        actions = ActionChains(driver)

        # 移动到起始位置（通过body元素定位）
        body = driver.find_element(By.TAG_NAME, 'body')
        actions.move_to_element(body).perform()  # 先重置鼠标位置

        # 计算偏移量（相对当前鼠标位置）
        start_offset_x = start_x
        start_offset_y = y_pos

        # 分解移动步骤（每0.1秒移动一次）
        steps = 20
        step_x = (end_x - start_x) / steps

        # 执行动作序列（使用相对移动）
        (actions
         .move_by_offset(start_offset_x, start_offset_y)  # 移动到起始点
         .click_and_hold()
         .pause(0.1))

        for _ in range(steps):
            actions.move_by_offset(step_x, 0).pause(0.1)

        (actions
         .release()
         .perform())

    except Exception as e:
        logging.error(f"perform_viewport_drag error: {e}")


def perform_mild_move():
    body = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    # 动作链：移动鼠标到窗口中心
    actions = ActionChains(driver)
    actions.move_to_element_with_offset(body, 0, 0).perform()

    # 再向右微移 1 像素
    actions.move_by_offset(random.randint(1, 3), random.randint(1, 3)).perform()


base_status_cv: Optional[np.ndarray] = None


def get_base_loading_state_img():
    global base_status_cv
    data: bytes = driver.get_screenshot_as_png()
    img = Image.open(BytesIO(data))
    width, height = img.size
    left, top, right, bottom = 0, height - 30, 160, height
    base_img = img.crop((left, top, right, bottom))
    base_status_cv = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)


def is_loading_complete():
    if base_status_cv is None:
        logging.error("base_status_cv is None, please call get_base_loading_state_img first")
        return False
    # 获取当前截图
    current_data = driver.get_screenshot_as_png()
    current_img = Image.open(BytesIO(current_data))
    width, height = current_img.size
    left, top, right, bottom = 0, height - 30, 160, height
    current_cropped = current_img.crop((left, top, right, bottom))
    current_cv = cv2.cvtColor(np.array(current_cropped), cv2.COLOR_RGB2BGR)

    # 计算差异
    diff = cv2.absdiff(base_status_cv, current_cv)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # 计算差异值
    diff_value = np.sum(diff_thresh) / 255  # 计算差异像素数量
    return diff_value < 5
