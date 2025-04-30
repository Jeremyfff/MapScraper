import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-8s %(asctime)-24s %(filename)-24s:%(lineno)-4d | %(message)s")
logging.getLogger("PIL").setLevel(logging.WARNING)  # Disable PIL's DEBUG output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time
from src.dataclasses import ScrappingConfig

from src import win_utils as wu, selenium_utils as su, rd_utils as ru
from src.web_scrapper import GoogleEarthWebScrapper


def init_driver():
    """Open and inject into driver"""
    driver = su.open_driver()
    time.sleep(1)
    popup_hwnd = wu.find_window_with_title(window_tile=["Google Chrome Gpu", "Chromium Gpu"])
    pid = wu.get_chrome_gpu_pid(popup_hwnd)
    ru.inject_into_process(pid)
    time.sleep(0.2)
    wu.close_chrome_gpu_popup(popup_hwnd)
    time.sleep(0.2)
    return driver


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', help='指定配置文件路径', default=None)
    parser.add_argument('-g', '--gap', help='捕捉间隔', default=1.0)
    parser.add_argument('-t', '--target', help='输出文件夹', default=None)

    _args = parser.parse_args()
    if _args.config:
        _config = ScrappingConfig.load_from_file(_args.config)
    else:
        _config = None
    _driver = init_driver()
    with ru.TargetControllerWrapper(_driver) as _target_control:
        GoogleEarthWebScrapper(_driver, _target_control, _config, _args).run()
