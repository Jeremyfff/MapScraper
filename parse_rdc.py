import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-8s %(asctime)-24s %(filename)-24s:%(lineno)-4d | %(message)s")
logging.getLogger("PIL").setLevel(logging.WARNING)  # Disable PIL's DEBUG output

import numpy as np
import open3d as o3d
from src.dataclasses import BatchData
from src.rd_utils import CaptureWrapper
from src.capture_parser import GoogleEarthCapturerParser, BaiduMapCapturerParser
from src.o3d_utils import create_open3d_meshes, create_coordinate_axis

np.set_printoptions(
    suppress=True,
    precision=10,
    floatmode='fixed'
)


def extract_data_from_rdc__google_earth(file_name: str, show_progress=False, print_duration=False, **kwargs) -> BatchData:
    with CaptureWrapper(file_name) as controller:
        cap_scraper = GoogleEarthCapturerParser(file_name, controller, **kwargs)
        extracted_data = cap_scraper.run(show_progress, print_duration)
    return extracted_data

def extract_data_from_rdc__baidu_map(file_name: str, show_progress=False, print_duration=False, **kwargs) -> BatchData:
    with CaptureWrapper(file_name) as controller:
        cap_scraper = BaiduMapCapturerParser(file_name, controller, **kwargs)
        extracted_data = cap_scraper.run(show_progress, print_duration)
    return extracted_data

if __name__ == "__main__":
    meshes = []
    # # use GoogleEarth
    # file_name = "./results/2025-04-29_22-31-12/rdc/1_1.rdc"
    # batch_data = extract_data_from_rdc__google_earth(file_name, show_progress=False, print_duration=True)

    # use BaiduMapa
    file_name = "./results/baidumap.rdc"
    batch_data = extract_data_from_rdc__baidu_map(file_name, show_progress=False, print_duration=True)

    meshes.extend(create_open3d_meshes(batch_data, show_progress=False, print_duration=True))
    meshes.append(create_coordinate_axis(size=5.0))
    o3d.visualization.draw_geometries(meshes)
