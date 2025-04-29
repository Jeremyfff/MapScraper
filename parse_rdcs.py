# -*- coding: utf-8 -*-
# @Author  : Yiheng Feng
# @Time    : 4/28/2025 7:52 PM
# @Function:
import logging
import os
import threading
import time
from collections import deque, defaultdict
from enum import IntFlag, auto

import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import KDTree
from parse_rdc import extract_data_from_rdc, BatchData, DrawData, create_open3d_meshes, create_open3d_mesh, create_coordinate_axis


class BatchHelperFlags(IntFlag):
    NONE = 0
    DISCARD_TEXTURE = auto()  # 0x00000001
    DISCARD_UVS = auto()  # 0x00000010
    DISCARD_ALL = auto()  # 0x00000100


class BatchHelper:
    def __init__(self, debug_geometries: list = None, flags: BatchHelperFlags = BatchHelperFlags.NONE):
        self.flags = flags
        self.batches: list[BatchData] = []
        self.batch_centers: list[np.ndarray] = []  # (nBatches, (nDraws, 3))
        self.batch_features: list[np.ndarray] = []  # (nBatches, (nDraws, 3))
        self.batch_transforms: list[np.ndarray] = []  # (nBatches, (3, ))
        self.batch_successes: list[bool] = []  # (nBatches, bool)
        self.debug_geometries = debug_geometries if debug_geometries is not None else []

    def get_success_batches(self) -> list[BatchData]:
        return [batch for batch, success in zip(self.batches, self.batch_successes) if success]

    @staticmethod
    def extract_draw_feature(draw: DrawData, texture: bool = True, uv: bool = True) -> np.ndarray:
        """提取形状特征，可选添加纹理和UV特征

        Args:
            draw: 包含顶点、纹理和UV数据的DrawData对象
            texture: 是否包含纹理特征（默认True）
            uv: 是否包含UV坐标特征（默认True）

        Returns:
            np.ndarray: 组合后的特征向量
        """
        # --------------------- 1. 形状特征 ---------------------
        centroid = np.mean(draw.vertices, axis=0)
        centered = draw.vertices - centroid
        cov_matrix = np.cov(centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues.sort()
        shape_feature = eigenvalues[::-1]  # (3,)

        # --------------------- 2. 可选特征 ---------------------
        feature_parts = [shape_feature]

        # 纹理特征（RGB均值）
        if texture and hasattr(draw, 'texture') and draw.texture is not None:
            texture_mean = np.mean(draw.texture[..., :3], axis=(0, 1))  # (3,)
            feature_parts.append(texture_mean)

        # UV坐标均值
        if uv and hasattr(draw, 'uvs') and draw.uvs is not None:
            uv_mean = np.mean(draw.uvs, axis=0)  # (2,)
            feature_parts.append(uv_mean)

        # --------------------- 3. 拼接特征 ---------------------
        combined_feature = np.concatenate(feature_parts)
        return combined_feature

    @staticmethod
    def extract_batch_features(batch: BatchData) -> np.ndarray:
        features = []
        for draw in batch.draw_datas.values():
            feature = BatchHelper.extract_draw_feature(draw)
            features.append(feature)
        return np.array(features)  # (nDraws, features)

    @staticmethod
    def extract_batch_centers(batch: BatchData) -> np.ndarray:
        centers = []
        for draw in batch.draw_datas.values():
            centroid = np.mean(draw.vertices, axis=0)
            centers.append(centroid)
        return np.array(centers)

    def addBatch(self, batch: BatchData):
        """add a batch and align to other batches"""
        if batch in self.batches:
            logging.error(f"Batch {batch} already exists")
            return

        # self._preprocess_batch(batch)
        self._align_batch_in_adding_stage(batch)  # move batch to align the other blocks
        self._post_process_batch_in_adding_stage(batch)
        self.batches.append(batch)

    def _preprocess_batch_in_adding_stage(self, batch: BatchData):
        """删除batch中大小不合适的块"""
        draws: dict[int, DrawData] = batch.draw_datas

        # 保存键和对应的包围盒，用于后续删除
        keys = []
        aabbs = []
        max_sizes = []

        # 遍历所有 draw 数据并计算包围盒
        for key, draw in draws.items():
            mesh = create_open3d_mesh(draw.vertices, draw.triangles, None, None)
            aabb = mesh.get_axis_aligned_bounding_box()

            # 计算最大尺寸
            size = aabb.get_max_bound() - aabb.get_min_bound()
            max_size = max(size)

            keys.append(key)
            aabbs.append(aabb)
            max_sizes.append(max_size)

        if not aabbs:
            return

        # 计算中位数和阈值
        median_size = np.median(max_sizes)
        threshold = 1.5 * median_size

        # 收集需要删除的键
        to_remove = [key for key, size in zip(keys, max_sizes) if size > threshold]

        # 从原始数据中删除异常项
        for key in to_remove:
            draws.pop(key)

        # 标记颜色（已删除的项仍会显示为红色）
        for aabb, size in zip(aabbs, max_sizes):
            aabb.color = (1, 0, 0) if size > threshold else (0, 1, 0)

        # 添加到调试几何体（包括已删除的项）
        # self.debug_geometries.extend(aabbs)

        # # 打印统计信息
        # print(f"原始包围盒数量: {len(aabbs)}")
        # print(f"移除异常项数量: {len(to_remove)}")
        # print(f"剩余有效项数量: {len(draws)}")
        # print(f"中位数尺寸: {median_size:.2f}, 阈值: {threshold:.2f}")

    def _align_batch_in_adding_stage(self, batch: BatchData, feature_vector_threshold=0.05, max_top_k=10) -> bool:
        features: np.ndarray = self.extract_batch_features(batch)
        centers: np.ndarray = self.extract_batch_centers(batch)
        avg_translation = np.zeros((3,), dtype=np.float32)
        assert len(features) == len(centers)
        if len(self.batch_features) == 0:
            self.batch_features.append(features)
            self.batch_centers.append(centers)
            self.batch_transforms.append(avg_translation)
            self.batch_successes.append(True)
            return True
        prev_cursor = len(self.batches)
        success = False

        while prev_cursor > 0:  # 从后往前搜索
            prev_cursor -= 1
            prev_features: np.ndarray = self.batch_features[prev_cursor]
            prev_centers: np.ndarray = self.batch_centers[prev_cursor]

            # --------------------- 1. 计算欧氏距离矩阵 ---------------------
            dist_matrix = np.sqrt(
                np.sum((features[:, np.newaxis] - prev_features[np.newaxis, :]) ** 2, axis=2)
            )
            # --------------------- 2. 建立初步匹配关系 ---------------------
            candidate_matches = []
            for i in range(len(features)):
                best_match_idx = np.argmin(dist_matrix[i])
                min_distance = dist_matrix[i, best_match_idx]
                if min_distance < feature_vector_threshold:
                    candidate_matches.append((i, best_match_idx, min_distance))

            # --------------------- 3. 筛选最佳匹配（距离最小的前10个） ---------------------
            # 按距离从小到大排序
            sorted_matches = sorted(candidate_matches, key=lambda x: x[2])

            # 取前10个匹配（如果总匹配数不足则取全部）
            max_matches = min(max_top_k, len(sorted_matches))
            best_matches = sorted_matches[:max_matches]
            # 提取匹配信息和距离
            matches = [(m[0], m[1]) for m in best_matches]
            distances = [m[2] for m in best_matches]

            # --------------------- 4. 计算平移量 ---------------------
            if len(matches) == 0:
                continue

            print([f"从{len(candidate_matches)}个match中筛选出前{len(best_matches)}个结果 -> {match}:{dist:.4f}" for match, dist in zip(matches, distances)])

            # 收集匹配项的中心点偏移
            # prev_drawcallIds = list(prev_batch.draw_datas.keys())
            # curr_drawcallIds = list(batch.draw_datas.keys())
            translation_vectors = []

            for curr_idx, prev_idx in matches:
                curr_centroid = centers[curr_idx]
                prev_centroid = prev_centers[prev_idx]
                translation_vectors.append(prev_centroid - curr_centroid)

            # --------------------- 5. 加权平移（使用前10个匹配的距离权重） ---------------------
            weights = 1.0 / (np.array(distances) + 1e-6)
            weights /= np.sum(weights)
            avg_translation = np.dot(weights, translation_vectors)

            # --------------------- 6. 应用平移 ---------------------
            for draw in batch.draw_datas.values():
                draw.vertices += avg_translation
            centers += avg_translation  # 同时移动centers
            # features与位置无关，不动
            success = True
            break
        if not success:
            logging.warning(f"Align failed after {len(self.batch_features)} attempts.batch file name: {batch.file_name}")
        self.batch_features.append(features)
        self.batch_centers.append(centers)
        self.batch_transforms.append(avg_translation)
        self.batch_successes.append(success)
        return success

    def _post_process_batch_in_adding_stage(self, batch: BatchData):
        if self.flags & BatchHelperFlags.DISCARD_UVS:
            for draw in batch.draw_datas.values():
                draw.uvs = None

        if self.flags & BatchHelperFlags.DISCARD_TEXTURE:
            for draw in batch.draw_datas.values():
                draw.texture = None

        if self.flags & BatchHelperFlags.DISCARD_ALL:
            for draw in batch.draw_datas.values():
                draw.vertices = None
                draw.triangles = None
                draw.uvs = None
                draw.texture = None
                draw.constants = None
        return batch

    def post_process_batches(self,
                             spatial_threshold=0.01,
                             feature_threshold=0.01,
                             top_k=10,
                             debug_bbox=False,
                             debug_print=False):
        """后处理批次数据，删除空间和特征相似的draws

        Args:
            spatial_threshold: 空间距离阈值
            feature_threshold: 特征相似度阈值
            top_k: 初始搜索数量
            debug_bbox: 是否生成被删除draw的包围盒
            debug_print: 是否打印处理日志
        """

        # --------------------- 阶段1: 过滤失败批次 ---------------------
        # 创建成功批次的索引列表
        success_indices = [i for i, success in enumerate(self.batch_successes) if success]
        original_batch_count = len(self.batches)

        # 更新所有数据为仅包含成功批次
        self.batches = [self.batches[i] for i in success_indices]
        self.batch_centers = [self.batch_centers[i] for i in success_indices]
        self.batch_features = [self.batch_features[i] for i in success_indices]
        self.batch_transforms = [self.batch_transforms[i] for i in success_indices]
        self.batch_successes = [self.batch_successes[i] for i in success_indices]
        if debug_print:
            logging.info(f"✅ 批次过滤: 原始批次 {original_batch_count} -> 成功批次 {len(self.batches)}")
        # --------------------- 阶段2: 展平所有draw数据 ---------------------
        all_draws = []  # 存储所有draw数据
        all_centers = []  # 存储所有中心点坐标
        all_features = []  # 存储所有特征向量
        draw_id_map = {}  # 映射draw索引到原始位置 {flat_idx: (batch_idx, drawcallId)}
        total_draws_before = 0
        # 遍历所有成功批次
        for batch_idx in range(len(self.batches)):
            batch = self.batches[batch_idx]
            centers = self.batch_centers[batch_idx]
            features = self.batch_features[batch_idx]
            total_draws_before += len(batch.draw_datas)
            # 验证数据一致性
            assert len(batch.draw_datas) == len(centers) == len(features)

            # 展平数据并建立映射
            for draw_idx, (drawcallId, draw_data) in enumerate(batch.draw_datas.items()):
                flat_idx = len(all_draws)
                all_draws.append(draw_data)
                all_centers.append(centers[draw_idx])
                all_features.append(features[draw_idx])
                draw_id_map[flat_idx] = (batch_idx, drawcallId)

        # 转换为numpy数组以提升性能
        all_centers = np.array(all_centers)  # (nTotalDraws, 3)
        all_features = np.array(all_features)  # (nTotalDraws, nFeatures)

        # --------------------- 阶段3: 构建KDTree进行空间搜索 ---------------------

        kdtree = KDTree(all_centers)

        # 记录需要删除的draw标记
        to_delete = set()

        # --------------------- 阶段4: 遍历所有draw进行相似性检测 ---------------------
        for current_idx in range(len(all_draws)):
            # 跳过已标记删除的draw
            if current_idx in to_delete:
                continue

            # 查询最近的top_k个draw（包含自己）
            distances, neighbors = kdtree.query(
                all_centers[current_idx],
                k=top_k,
                distance_upper_bound=spatial_threshold * 2  # 初步过滤
            )

            # 筛选有效邻居（距离在阈值范围内）
            valid_neighbors = [
                n for n, d in zip(neighbors, distances)
                if d <= spatial_threshold and n < len(all_draws)
            ]

            # 检查特征相似性
            similar_draws = []
            for neighbor_idx in valid_neighbors:
                # 跳过自己
                if neighbor_idx == current_idx:
                    continue

                # 计算特征距离
                feature_dist = np.linalg.norm(
                    all_features[current_idx] - all_features[neighbor_idx]
                )

                if feature_dist <= feature_threshold:
                    similar_draws.append(neighbor_idx)
            # --------------------- 阶段5: 标记需要删除的draw ---------------------
            # 保留第一个出现的draw，删除后续重复项
            if similar_draws:
                # 按出现顺序排序（flat_idx小的先出现）
                sorted_similar = sorted([current_idx] + similar_draws)

                # 保留第一个，标记其余为删除
                for delete_idx in sorted_similar[1:]:
                    to_delete.add(delete_idx)

        # --------------------- 阶段6: 将删除标记映射回原始batch结构 ---------------------
        # 构建每个batch的删除集合 {batch_idx: set(draw_ids)}
        batch_delete_map = defaultdict(set)

        for delete_idx in to_delete:
            batch_idx, drawcallId = draw_id_map[delete_idx]
            batch_delete_map[batch_idx].add(drawcallId)

        # --------------------- 阶段7: 执行批量删除操作 ---------------------

        for batch_idx in range(len(self.batches)):
            # 获取当前batch数据
            batch = self.batches[batch_idx]
            delete_ids = batch_delete_map.get(batch_idx, set())

            if not delete_ids:
                continue
            # ===================== 新增调试几何体生成 =====================
            # 遍历需要删除的draw生成包围盒
            if debug_bbox:
                for drawcallId in delete_ids:
                    draw = batch.draw_datas[drawcallId]  # 获取将被删除的draw数据

                    # 生成调试几何体
                    if draw.vertices is not None and draw.triangles is not None:
                        mesh = create_open3d_mesh(draw.vertices, draw.triangles, None, None)
                        aabb = mesh.get_axis_aligned_bounding_box()
                        aabb.color = (1, 0, 0)
                        self.debug_geometries.append(aabb)
            # ===================== 调试代码结束 =====================

            # 删除指定draws
            new_draws = {
                drawcallId: draw_data
                for drawcallId, draw_data in batch.draw_datas.items()
                if drawcallId not in delete_ids
            }

            # 更新batch数据
            batch.draw_datas = new_draws

            # 更新缓存数据
            keep_indices = [
                idx for idx, drawCallId in enumerate(batch.draw_datas.keys())
                if drawCallId not in delete_ids
            ]
            self.batch_centers[batch_idx] = self.batch_centers[batch_idx][keep_indices]
            self.batch_features[batch_idx] = self.batch_features[batch_idx][keep_indices]
        # 打印汇总信息
        if debug_print:
            total_draws_after = sum(len(batch.draw_datas) for batch in self.batches)
            logging.info("📊 ")
            logging.info(f"📊处理结果汇总:原始Draw总数({total_draws_before}) -> 最终Draw总数({total_draws_after})")


def extract_data_from_multiple_rdc(file_paths: list[str], debug_geometries: list = None,
                                   flags: BatchHelperFlags = BatchHelperFlags.NONE) -> list[BatchData]:
    assert len(file_paths) > 0, "No file paths provided"
    helper = BatchHelper(debug_geometries, flags)

    batches_queue: deque[BatchData] = deque()
    _read_complete = False

    ref_matrix = None
    def rdc_reader():
        nonlocal _read_complete
        for file_path in file_paths:
            while len(batches_queue) > 3:
                time.sleep(0.1)
            try:
                batches_queue.append(extract_data_from_rdc(file_path, show_progress=False, print_duration=False))
            except Exception as e:
                logging.error(f"Extract_data_from_rdc failed: {file_path}, reason: {str(e)}")
                continue
        _read_complete = True

    def rdc_parser():
        while True:
            if len(batches_queue) == 0 and _read_complete:
                break
            while len(batches_queue) == 0:
                time.sleep(0.1)
            batch = batches_queue.popleft()
            try:
                helper.addBatch(batch)
            except Exception as e:
                logging.error(f"Add batch failed: {batch.file_name}, reason: {str(e)}")
                continue
            logging.info(f"=====================Batch {batch.file_name} processed=========================")

    rdc_reader_thread = threading.Thread(target=rdc_reader)
    rdc_parser_thread = threading.Thread(target=rdc_parser)

    rdc_reader_thread.start()
    rdc_parser_thread.start()

    rdc_reader_thread.join()
    rdc_parser_thread.join()

    helper.post_process_batches(debug_bbox=False, debug_print=True)
    return helper.get_success_batches()


if __name__ == "__main__":

    folder_path = "./results/2025-04-29_22-31-12/rdc"  # 2025-04-27_19-18-14
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    file_paths.sort(key=lambda x: os.path.getctime(x))  # 按创建日期排序

    map_geometries = []
    debug_geometries = []

    debug_geometries.append(create_coordinate_axis(size=10.0))
    batch_datas: list[BatchData] = extract_data_from_multiple_rdc(file_paths, debug_geometries, flags=BatchHelperFlags.NONE)

    for batch_data in tqdm(batch_datas, desc="Creating Open3D Meshes"):
        map_geometries.extend(create_open3d_meshes(batch_data))

    logging.info("Starting Open3D")
    all_geometries = debug_geometries + map_geometries
    o3d.visualization.draw_geometries(all_geometries)
