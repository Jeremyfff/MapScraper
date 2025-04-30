import json
import os.path
from typing import Optional

import numpy as np
import open3d as o3d


class ScrappingConfig:
    def __init__(self):
        self.start_lat = 30
        self.end_lat = 30
        self.start_lon = 120
        self.end_lon = 120
        self.fov = 36
        self.dis = 800
        self.alt = 0
        self.lat_step = 0.01
        self.lon_step = 0.01

    @property
    def num_lat(self):
        return int((self.end_lat - self.start_lat) / self.lat_step) + 1

    @property
    def num_lon(self):
        return int((self.end_lon - self.start_lon) / self.lon_step) + 1

    def save(self, file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def load_from_file(config_path):
        config = ScrappingConfig()
        with open(config_path, 'r') as f:
            c = json.load(f)

        config.start_lat, config.start_lon = c['start_lat'], c['start_lon']
        config.end_lat, config.end_lon = c['end_lat'], c['end_lon']
        config.fov, config.dis, config.alt = c['fov'], c['dis'], c['alt']
        config.lat_step, config.lon_step = c["lat_step"], c["lon_step"]
        return config


class DrawData:
    def __init__(self, vertices, triangles, vertex_colors=None, vertex_normals=None, uvs=None, texture=None, constants=None):
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_colors = vertex_colors
        self.vertex_normals = vertex_normals
        self.uvs = uvs
        self.texture = texture
        self.constants = constants

    def extract_feature(self, contain_texture: bool = True, contain_uv: bool = True) -> np.ndarray:
        # --------------------- 1. 形状特征 ---------------------
        centroid = np.mean(self.vertices, axis=0)
        centered = self.vertices - centroid
        cov_matrix = np.cov(centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues.sort()
        shape_feature = eigenvalues[::-1]  # (3,)

        # --------------------- 2. 可选特征 ---------------------
        feature_parts = [shape_feature]

        # 纹理特征（RGB均值）
        if contain_texture and hasattr(self, 'texture') and self.texture is not None:
            texture_mean = np.mean(self.texture[..., :3], axis=(0, 1))  # (3,)
            feature_parts.append(texture_mean)

        # UV坐标均值
        if contain_uv and hasattr(self, 'uvs') and self.uvs is not None:
            uv_mean = np.mean(self.uvs, axis=0)  # (2,)
            feature_parts.append(uv_mean)

        # --------------------- 3. 拼接特征 ---------------------
        combined_feature = np.concatenate(feature_parts)
        return combined_feature  # (n, )

    def get_aabb(self):
        if self.vertices is None or self.triangles is None:
            return None
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        aabb = mesh.get_axis_aligned_bounding_box()
        return aabb

    def translate(self, delta_translation: np.ndarray):
        self.vertices += delta_translation


class BatchData:
    def __init__(self, file_name=""):
        self.file_name = file_name
        self.draw_datas: dict[int, DrawData] = {}  # {drawcallID: DrawData}

        self._features: Optional[np.ndarray] = None  # (nDraws, nFeatures)
        self._centers: Optional[np.ndarray] = None  # (nDraws, 3)

        self._translation: Optional[np.ndarray] = np.zeros((3,), dtype=np.float32)
        self._success = False  # bool

    def addDrawData(self, drawcallId: int, data: DrawData):
        self.draw_datas[drawcallId] = data

    @property
    def features(self) -> np.ndarray:
        if self._features is not None and len(self._features) != len(self.draw_datas):
            self._features = None
        if self._features is None:
            features = []
            for draw in self.draw_datas.values():
                feature = draw.extract_feature()
                features.append(feature)
            self._features = np.array(features)
        return self._features

    @property
    def centers(self) -> np.ndarray:
        if self._centers is not None and len(self._centers) != len(self.draw_datas):
            self._centers = None
        if self._centers is None:
            centers = []
            for draw in self.draw_datas.values():
                centroid = np.mean(draw.vertices, axis=0)
                centers.append(centroid)
            self._centers = np.array(centers)
        return self._centers

    @property
    def translation(self) -> np.ndarray:
        return self._translation

    @translation.setter
    def translation(self, value: np.ndarray):
        assert len(value) == 3
        translation_delta = value - self.translation
        self.translate(translation_delta)

    def translate(self, delta_translation: np.ndarray):
        assert len(delta_translation) == 3
        self._translation += delta_translation
        if self._centers is not None:
            self._centers += delta_translation
        for draw in self.draw_datas.values():
            draw.translate(delta_translation)

    @property
    def success(self):
        return self._success

    @success.setter
    def success(self, value: bool):
        self._success = value
