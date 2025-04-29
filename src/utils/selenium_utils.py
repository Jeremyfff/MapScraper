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


def get_data_from_current_url(max_retry_count=5) -> str:
    assert driver is not None, "driver is None, please open it first"
    data = ''
    for i in range(max_retry_count):
        try:
            data = driver.current_url.split("/data=")[1]
            break
        except Exception as e:
            logging.warning(f"第{i + 1}/{max_retry_count}次尝试，data获取失败, 原因: {e}")
        time.sleep(1)
    if not data:
        logging.error("data获取失败")
    return data


def parse_google_earth_url(url: str) -> Dict[str, Optional[float]]:
    """
    从Google Earth URL中解析出相机参数

    参数:
        url: 完整的Google Earth URL，例如:
            "https://earth.google.com/web/@25.05531592,121.48961925,2.87784854a,868.45101958d,36y,2.73906771h,0t,0r/data=..."

    返回:
        包含解析参数的字典，键名与生成函数参数一致
    """
    # 初始化结果字典，所有值默认为None
    result = {
        'latitude': None,
        'longitude': None,
        'altitude': None,
        'distance': None,
        'fov': None,
        'heading': None,
        'tilt': None,
        'roll': None,
        'data': None
    }

    # 匹配参数部分的正则表达式
    # 示例匹配: /@25.05531592,121.48961925,2.87784854a,868.45101958d,36y,2.73906771h,0t,0r
    pattern = r"web/@(?P<params>[^/]+)"
    match = re.search(pattern, url)
    if not match:
        return result

    # 匹配data部分的正则表达式
    data_pattern = r"data=(?P<data>[^&]+)"
    data_match = re.search(data_pattern, url)
    if data_match:
        result['data'] = data_match.group('data')

    # 分割参数部分
    params_str = match.group('params')
    param_parts = params_str.split(',')

    if len(param_parts) < 8:
        return result

    try:
        # 解析各个参数
        result['latitude'] = float(param_parts[0])
        result['longitude'] = float(param_parts[1])
        result['altitude'] = float(param_parts[2].rstrip('a'))
        result['distance'] = float(param_parts[3].rstrip('d'))
        result['fov'] = float(param_parts[4].rstrip('y'))
        result['heading'] = float(param_parts[5].rstrip('h'))
        result['tilt'] = float(param_parts[6].rstrip('t'))
        result['roll'] = float(param_parts[7].rstrip('r'))
    except (ValueError, IndexError, AttributeError):
        # 如果解析失败，保持None值
        pass

    return result


def generate_google_earth_path(
        latitude: float,
        longitude: float,
        altitude: float = 0,
        distance: float = 800,
        fov: float = 36,
        heading: float = 0,
        tilt: float = 0,
        roll: float = 0,
        data: str = None
) -> str:
    """
    生成Google Earth的URL路径部分，参数说明：
    - latitude:  纬度 (25.05531592)
    - longitude: 经度 (121.48961925)
    - altitude:  海拔高度 (2.87784854)
    - distance:  相机到地面的距离 (868.45101958)
    - fov:       视野角度 (36)
    - heading:   水平旋转角 (2.73906771)
    - tilt:      俯仰角 (0)
    - roll:      翻滚角 (0)
    - data:      可通过get_data_from_current_url获得
    """
    params = [
        f"{latitude:.8f}",  # 纬度
        f"{longitude:.8f}",  # 经度
        f"{altitude:.8f}a",  # 海拔高度 + 'a'
        f"{distance:.8f}d",  # 地面距离 + 'd'
        f"{fov:.8f}y",  # FOV角度 + 'y'
        f"{heading:.8f}h",  # 水平旋转角 + 'h'
        f"{tilt:.8f}t",  # 俯仰角 + 't'
        f"{roll:.8f}r"  # 翻滚角 + 'r'
    ]
    if data is None:
        data = get_data_from_current_url()
    data_part = f'/data={data}' if data else ''
    # 拼接URL路径
    path = f"/web/@{','.join(params)}{data_part}"
    return path


def update_state(path: str):
    driver.execute_script(f"window.history.pushState({{}}, '', '{path}');")
    driver.execute_script("window.dispatchEvent(new PopStateEvent('popstate'));")


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
