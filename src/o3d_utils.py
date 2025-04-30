# -*- coding: utf-8 -*-
# @Author  : Yiheng Feng
# @Time    : 4/30/2025 9:56 AM
# @Function:
import logging
import time
from typing import Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.dataclasses import BatchData


def create_open3d_mesh(vertices, triangles, uvs, texture) -> Optional[o3d.geometry.TriangleMesh]:
    """创建带纹理的Open3D网格"""
    if vertices is None or len(vertices) == 0 or triangles is None or len(triangles) == 0:
        return None
    mesh = o3d.geometry.TriangleMesh()
    # 设置顶点和三角形
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    # 处理UV坐标
    if uvs is not None:
        uvs[:, 1] = 1 - uvs[:, 1]  # 翻转V方向
        # 按三角形顶点顺序重组UV
        triangle_uvs = uvs[triangles.reshape(-1)]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)

    # 加载纹理
    if texture is not None:
        mesh.textures = [o3d.geometry.Image(texture)]
        material_ids = np.zeros(len(triangles), dtype=np.int32)
        mesh.triangle_material_ids = o3d.utility.IntVector(material_ids)
    return mesh


def create_open3d_meshes(batch_data: "BatchData", show_progress=False, print_duration=False) -> list[o3d.geometry.TriangleMesh]:
    out_meshes = []
    draw_datas = batch_data.draw_datas
    start_time = time.time()
    for drawcall_id, data in tqdm(draw_datas.items(), total=len(draw_datas), desc="Creating Open3D Meshes", disable=not show_progress):
        mesh = create_open3d_mesh(data.vertices, data.triangles, data.uvs, data.texture)
        if mesh is not None:
            out_meshes.append(mesh)
    duration = time.time() - start_time
    if print_duration:
        logging.info(f"Created {len(out_meshes)} meshes in {duration:.2f}s")
    return out_meshes


def create_coordinate_axis(origin=None, size=1.0):
    """
    创建表示坐标轴的 LineSet
    参数:
        origin (list): 坐标原点 [x, y, z]
        size (float): 坐标轴长度
    返回:
        o3d.geometry.LineSet: 包含 X/Y/Z 轴的线段集合
    """
    origin = [0, 0, 0] if origin is None else origin
    # 定义坐标轴端点
    points = [
        origin,  # 原点
        [origin[0] + size, origin[1], origin[2]],  # X 轴
        [origin[0], origin[1] + size, origin[2]],  # Y 轴
        [origin[0], origin[1], origin[2] + size]  # Z 轴
    ]

    # 定义线段连接关系（原点到各轴端点）
    lines = [
        [0, 1],  # X 轴
        [0, 2],  # Y 轴
        [0, 3]  # Z 轴
    ]

    # 定义颜色（X: 红, Y: 绿, Z: 蓝）
    colors = [
        [1, 0, 0],  # 红色
        [0, 1, 0],  # 绿色
        [0, 0, 1]  # 蓝色
    ]

    # 创建 LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set
