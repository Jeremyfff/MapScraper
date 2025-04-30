import abc
import datetime
import json
import logging
import math
import os
import os.path
import re
import shutil
import threading
import time

from src import selenium_utils as su, rd_utils as ru
from src.dataclasses import ScrappingConfig
from src.rd_utils import rd

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class BaseWebScrapper:
    def __init__(self, driver, target_control, config, args):
        self.driver = driver
        self.target_control = target_control
        self.args = args
        self.config = config

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    @abc.abstractmethod
    def parse_url(self, url: str) -> dict[str: any]:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_path(self, **kwargs) -> str:
        raise NotImplementedError

    def update_state(self, path: str):
        self.driver.execute_script(f"window.history.pushState({{}}, '', '{path}');")
        self.driver.execute_script("window.dispatchEvent(new PopStateEvent('popstate'));")


class GoogleEarthWebScrapper(BaseWebScrapper):

    def run(self):
        print(f"=" * 100)
        # Stage1: Loading and setting
        self.driver.get("https://earth.google.com")
        input(f"\n等待页面加载完毕，手动进入地图界面。"
              f"推荐关闭地名图层，并在设置中关闭飞行动画, 将画质调到最高。\n"
              f"确保左下方的加载进度为100%，准备好后按回车继续。\n"
              f"{'=' * 100}\n")
        su.get_base_loading_state_img()

        if self.config is None:
            logging.info("Start building config...")
            self.config = self._build_config()
        assert self.config is not None

        # Stage2: Scrapping
        config = self.config  # to be more concise
        args = self.args
        self.update_state(self.generate_path(config.start_lat, config.start_lon, config.alt, config.dis, config.fov, 0, 0, 0))
        for i in range(5):
            logger.info(f"程序将在{5 - i}s后开始")
            time.sleep(1)

        _stop_move = False
        project_folder = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" if args.target is None else args.target
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
                    self.update_state(self.generate_path(config.start_lat + i * config.lat_step, config.start_lon + j * config.lon_step, config.alt, config.dis, config.fov, 0, 0, 0))
                    time.sleep(args.gap)  # 等待移动与加载完成
                    while True:
                        if su.is_loading_complete():
                            time.sleep(0.1)
                            break
                        logger.info("waiting for loading complete...")
                        time.sleep(0.5)
                    logger.info(f"{i}_{j}: start captrue")
                    self.driver.save_screenshot(os.path.join(snapshots_folder, f"{i}_{j}.png"))
                    data = self.parse_url(self.driver.current_url)
                    ru.trigger_capture(self.target_control, lambda capture_data: on_capture_complete(capture_data, data))
                    while 'complete' not in data or not data['complete']:
                        su.perform_mild_move()
                        time.sleep(0.1)
                    data.pop('complete')
                    src: str = str(data['path'])
                    dst: str = str(os.path.abspath(os.path.join(rdc_folder, f"{i}_{j}.rdc")))
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

    def _build_config(self) -> ScrappingConfig:
        time.sleep(1)
        input("请移动到要爬取区域的其中一个边角, 完成后按回车继续")
        start_pos = self.parse_url(self.driver.current_url)
        start_lat, start_lon = start_pos["latitude"], start_pos["longitude"]
        print(f"边角1: {start_lat}, {start_lon}")
        input("请移动到要爬取区域的另一个对角点, 完成后按回车继续")
        end_pos = self.parse_url(self.driver.current_url)
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
                self.update_state(self.generate_path(start_lat, start_lon, alt, dis, fov, 0, 0, 0))
                if input("确认当前精度? [y/n]") == 'y':
                    break
            input("请轻微拖动相机, 完成后按回车")
            grid_start = self.parse_url(self.driver.current_url)
            input("请对角线平移视口至（X+1, Y+1）的格网，视口可以有部分重叠，作为爬取时的重叠部分, 完成后按回车")
            grid_end = self.parse_url(self.driver.current_url)
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
                    config.save(file_name)
                    logger.info(f"配置文件已保存至{os.path.abspath(file_name)}")
                except Exception as e:
                    logging.error(f"保存配置文件失败: {e}")
        return config

    def _get_data_from_current_url(self, max_retry_count=5) -> str:
        assert self.driver is not None, "driver is None, please open it first"
        data = ''
        for i in range(max_retry_count):
            try:
                data = self.driver.current_url.split("/data=")[1]
                break
            except Exception as e:
                logging.warning(f"第{i + 1}/{max_retry_count}次尝试，data获取失败, 原因: {e}")
            time.sleep(1)
        if not data:
            logging.error("data获取失败")
        return data

    def parse_url(self, url: str) -> dict[str: any]:
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

    def generate_path(self, latitude: float, longitude: float, altitude: float = 0, distance: float = 800, fov: float = 36, heading: float = 0, tilt: float = 0, roll: float = 0, data: str = None) -> str:
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
            data = self._get_data_from_current_url()
        data_part = f'/data={data}' if data else ''
        # 拼接URL路径
        path = f"/web/@{','.join(params)}{data_part}"
        return path
