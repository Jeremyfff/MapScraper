import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-8s %(asctime)-24s %(filename)-24s:%(lineno)-4d | %(message)s")
logging.getLogger("PIL").setLevel(logging.WARNING)  # Disable PIL's DEBUG output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import datetime
import json
import math
import os
import os.path
import shutil
import threading
import time
from typing import Callable
from src.dataclasses import ScrappingConfig

from src.renderdoc_importer import rd
from src.utils import win_utils as wu
from src.utils import selenium_utils as su


def inject_into_process(pid, capture_options: rd.CaptureOptions = None):
    assert pid > 0, "PID must be greater than 0"
    if capture_options is None:
        capture_options = rd.GetDefaultCaptureOptions()
    result = rd.InjectIntoProcess(pid, [], os.path.join(os.path.abspath('.') + "/"), capture_options, False)
    if result.status == rd.ReplayStatus.Succeeded:
        logging.info("Injection Success! Now you can close the PID Popup Window")
    else:
        raise Exception(f"Injection Failed! status: {result.status}")


def create_target_control() -> rd.TargetControl:
    next_ident = 0
    idents = []
    while True:
        next_ident = rd.EnumerateRemoteTargets("", next_ident)
        if next_ident:
            idents.append(next_ident)
        else:
            break
    assert idents, "No Remote Targets Found"
    ident = idents[0]
    target_control: rd.TargetControl = rd.CreateTargetControl("", ident, "my_client", True)
    assert target_control is not None, "Create Failed"
    assert target_control.Connected()
    logging.info("Target Control Connected")
    print_target_control_info(target_control)
    return target_control


def print_target_control_info(target_control: rd.TargetControl):
    logging.info(f"Connected: {target_control.Connected()}")
    logging.info(f"API: {target_control.GetAPI()}")
    logging.info(f"BusyClient: {target_control.GetBusyClient()}")
    logging.info(f"PID: {target_control.GetPID()}")
    logging.info(f"Target: {target_control.GetTarget()}")


_in_capture_thread = False


def trigger_capture(target_control: rd.TargetControl, success_callback: Callable[[rd.NewCaptureData], None], timeout=60):
    global _in_capture_thread
    if _in_capture_thread:
        logging.warning("Capture is already in progress, return")
        return
    assert target_control.Connected(), "target_control has disconnected"
    target_control.TriggerCapture(1)

    def _msg_handle_thread():
        global _in_capture_thread
        _in_capture_thread = True
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = target_control.ReceiveMessage(None)
            if msg.type == 4:
                success_callback(msg.newCapture)
                _in_capture_thread = False
                return
        _in_capture_thread = False
        raise Exception("Capture Timeout")

    threading.Thread(target=_msg_handle_thread).start()


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Google Earth路径生成工具')
    parser.add_argument('-c', '--config', help='指定配置文件路径', default=None)
    parser.add_argument('-g', '--gap', help='捕捉间隔', default=1.0)
    parser.add_argument('--base_url', help='访问的网址', default="https://earth.google.com")

    args = parser.parse_args()

    if args.config:
        config = ScrappingConfig.load_from_file(args.config)
    else:
        config = None
    driver = su.open_driver()
    time.sleep(1)
    popup_hwnd = wu.find_window_with_title(window_tile="Google Chrome Gpu")
    pid = wu.get_chrome_gpu_pid(popup_hwnd)
    inject_into_process(pid)
    time.sleep(0.2)
    wu.close_chrome_gpu_popup(popup_hwnd)
    time.sleep(0.2)
    target_control = create_target_control()

    logging.info(f"=" * 100)
    driver.get(args.base_url)
    input("等待页面加载完毕，手动进入地图界面。推荐关闭地名图层，并在设置中关闭飞行动画。\n确保左下方的加载进度为100%，准备好后按回车继续")
    su.get_base_loading_state_img()

    if config is None:
        input("请移动到要爬取区域的其中一个边角, 完成后按回车继续")
        start_pos = su.parse_google_earth_url(driver.current_url)
        start_lat, start_lon = start_pos["latitude"], start_pos["longitude"]
        print(f"边角1: {start_lat}, {start_lon}")
        input("请移动到要爬取区域的另一个对角点, 完成后按回车继续")
        end_pos = su.parse_google_earth_url(driver.current_url)
        end_lat, end_lon = end_pos["latitude"], end_pos["longitude"]
        print(f"边角2: {end_lat}, {end_lon}")
        if start_lat > end_lat:
            start_lat, end_lat = end_lat, start_lat
        if start_lon > end_lon:
            start_lon, end_lon = end_lon, start_lon

        num_lat, num_lon, lat_step, lon_step = 0, 0, 0, 0
        while True:
            while True:
                grid_size = int(input("请输入爬取的网格大小(m): "))
                fov = int(input("请输入爬取的视场角(度): "))
                dis = grid_size / 2 / math.tan(math.radians(fov / 2))
                alt = int(input("请输入当地的海拔(m):"))
                su.update_state(su.generate_google_earth_path(start_lat, start_lon, alt, dis, fov, 0, 0, 0))
                if input("确认当前精度? [y/n]") == 'y':
                    break
            input("请轻微拖动相机, 完成后按回车")
            grid_start = su.parse_google_earth_url(driver.current_url)
            input("请对角线平移视口至（X+1, Y+1）的格网，视口可以有部分重叠，作为爬取时的重叠部分, 完成后按回车")
            grid_end = su.parse_google_earth_url(driver.current_url)
            grid_start_lat, grid_start_lon = grid_start["latitude"], grid_start["longitude"]
            grid_end_lat, grid_end_lon = grid_end["latitude"], grid_end["longitude"]
            lat_step = abs(grid_start_lat - grid_end_lat)
            lon_step = abs(grid_start_lon - grid_end_lon)
            print(f"lat_step: {lat_step: .4f}, lon_step: {lon_step}")
            if lat_step == 0 or lon_step == 0:
                print("请重试")
                continue
            num_lat = int((end_lat - start_lat) / lat_step) + 1
            num_lon = int((end_lon - start_lon) / lon_step) + 1
            print(f"num_lat: {num_lat}, num_lon: {num_lon}, 共计{num_lat * num_lon}个格网点， 预计文件大小: {num_lat * num_lon * 45}MiB")
            if input("确认当前参数? [y/n]") == "y":
                break
        config = ScrappingConfig()
        config.start_lat, config.start_lon = start_lat, start_lon
        config.end_lat, config.end_lon = end_lat, end_lon
        config.fov, config.alt, config.dis = fov, alt, dis
        config.lat_step, config.lon_step = lat_step, lon_step
        if input("保存配置文件? [y/n]") == 'y':
            file_name = input("请输入文件名称, 以.json结尾")
            if file_name:
                try:
                    config.save_to_file(file_name)
                    logger.info(f"配置文件已保存至{os.path.abspath(file_name)}")
                except Exception as e:
                    logging.error(f"保存配置文件失败: {e}")
    assert config is not None

    su.update_state(su.generate_google_earth_path(config.start_lat, config.start_lon, config.alt, config.dis, config.fov, 0, 0, 0))
    for i in range(5):
        logger.info(f"程序将在{5 - i}s后开始")
        time.sleep(1)

    _stop_move = False
    project_folder = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    snapshots_folder = os.path.join(project_folder, "snapshots")
    os.makedirs(snapshots_folder, exist_ok=True)
    meta_folder = os.path.join(project_folder, "meta")
    os.makedirs(meta_folder, exist_ok=True)
    rdc_folder = os.path.join(project_folder, "rdc")
    os.makedirs(rdc_folder, exist_ok=True)


    def on_capture_complete(capture_data: "rd.NewCaptureData", data: dict):
        data['complete'] = True
        data['api'] = capture_data.api
        data['byteSize'] = capture_data.byteSize
        data['path'] = capture_data.path
        data['captureId'] = capture_data.captureId


    def move_thread():
        for i in range(config.num_lat):
            _range = list(range(config.num_lon))
            if i % 2 == 1:
                _range.reverse()
            for j in _range:
                if _stop_move:
                    return
                su.update_state(su.generate_google_earth_path(config.start_lat + i * config.lat_step, config.start_lon + j * config.lon_step, config.alt, config.dis, config.fov, 0, 0, 0))
                time.sleep(args.gap)  # 等待移动与加载完成
                while True:
                    if su.is_loading_complete():
                        time.sleep(0.1)
                        break
                    logger.info("waiting for loading complete...")
                    time.sleep(0.5)
                logger.info(f"{i}_{j}: start captrue")
                driver.save_screenshot(os.path.join(snapshots_folder, f"{i}_{j}.png"))
                data = su.parse_google_earth_url(driver.current_url)
                trigger_capture(target_control, lambda capture_data: on_capture_complete(capture_data, data))
                while 'complete' not in data or not data['complete']:
                    su.perform_mild_move()
                    time.sleep(0.1)
                data.pop('complete')
                src: str = str(data['path'])
                dst: str = os.path.abspath(os.path.join(rdc_folder, f"{i}_{j}.rdc"))
                logger.info(f"Moving {src} to {dst}")
                shutil.move(src, dst)
                data['path'] = dst
                with open(os.path.join(meta_folder, f"{i}_{j}.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                logger.info(f"{i}_{j} done. ")
        logging.info("All Complete ! PRESS ANY KEY TO EXIT")


    threading.Thread(target=move_thread).start()
    input("PRESS ANY KEY TO EXIT")
    _stop_move = True
