# -*- coding: utf-8 -*-
# @Author  : Yiheng Feng
# @Time    : 4/27/2025 5:34 PM
# @Function:
import logging
import time
from typing import Optional

import cv2
import open3d as o3d
import pyrr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-8s %(asctime)-24s %(filename)-24s:%(lineno)-4d | %(message)s")
logging.getLogger("PIL").setLevel(logging.WARNING)  # Disable PIL's DEBUG output
import struct
import numpy as np
import texture2ddecoder
from src.renderdoc_importer import rd

MATRIX_NAME = '_uMeshToWorldMatrix'
np.set_printoptions(
    suppress=True,  # 禁用科学计数法
    precision=10,  # 保留2位小数
    floatmode='fixed'  # 确保小数点后位数固定
)


class CaptureWrapper:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.cap = rd.OpenCaptureFile()
        status = self.cap.OpenFile(self.filename, '', None)

        if status != rd.ReplayStatus.Succeeded:
            logging.error("Couldn't open file: " + str(status))
            self.err = True
            return None

        if not self.cap.LocalReplaySupport():
            logging.error("Capture cannot be replayed")
            self.err = True
            return None

        status, self.controller = self.cap.OpenCapture(rd.ReplayOptions(), None)

        if status != rd.ReplayStatus.Succeeded:
            logging.error("Couldn't initialise replay: " + str(status))
            if status == 15:
                logging.error("This is likely due to an unsupported version of RenderDoc.")
            self.cap.Shutdown()
            self.err = True
            return None
        return self.controller

    def __exit__(self, type, value, traceback):
        self.controller.Shutdown()
        self.cap.Shutdown()


class MeshData(rd.MeshFormat):
    TYPE_MAP = {
        'B': np.uint8, 'b': np.int8,
        'H': np.uint16, 'h': np.int16,
        'I': np.uint32, 'i': np.int32,
        'L': np.uint64, 'l': np.int64,
        'e': np.float16, 'f': np.float32,
        'd': np.float64
    }

    def __init__(self, attr, ib, vbs, draw):
        super().__init__()
        self._build(attr, ib, vbs, draw)

    def _build(self, attr, ib, vbs, draw):
        # We don't handle instance attributes
        if attr.perInstance:
            raise RuntimeError("Instanced properties are not supported!")
        self.indexResourceId = ib.resourceId
        self.indexByteOffset = ib.byteOffset
        self.indexByteStride = ib.byteStride
        self.baseVertex = draw.baseVertex
        self.indexOffset = draw.indexOffset
        self.numIndices = draw.numIndices

        # If the draw doesn't use an index buffer, don't use it even if bound
        if not (draw.flags & rd.DrawFlags.Indexed):
            self.indexResourceId = rd.ResourceId.Null()

        # The total offset is the attribute offset from the base of the vertex
        self.vertexByteOffset = attr.byteOffset + vbs[attr.vertexBuffer].byteOffset
        self.format = attr.format
        self.vertexResourceId = vbs[attr.vertexBuffer].resourceId
        self.vertexByteStride = vbs[attr.vertexBuffer].byteStride
        self.name = attr.name
        self.format_str = self.fmt2str(self.format)
        self._has_built = True

    @staticmethod
    def fmt2str(fmt) -> str:
        """Convert fmt format specification to struct-style format char"""
        if fmt.Special():
            raise RuntimeError("Packed formats are not supported!")
        #           012345678
        uint_fmt = "xBHxIxxxL"
        sint_fmt = "xbhxixxxl"
        float_fmt = "xxexfxxxd"  # only 2, 4 and 8 are valid
        formatChars = {
            rd.CompType.UInt: uint_fmt,
            rd.CompType.SInt: sint_fmt,
            rd.CompType.Float: float_fmt,
            rd.CompType.UNorm: uint_fmt,
            rd.CompType.UScaled: uint_fmt,
            rd.CompType.SNorm: sint_fmt,
            rd.CompType.SScaled: sint_fmt,
        }
        return f"{fmt.compCount}{formatChars[fmt.compType][fmt.compByteWidth]}"

    def fetchIndices(self, controller):
        if self.indexResourceId != rd.ResourceId.Null():
            # 映射字节步长到对应的NumPy数据类型
            dtype_map = {2: np.uint16, 4: np.uint32}
            dtype = dtype_map.get(self.indexByteStride, np.uint8)

            # 获取索引缓冲区数据
            ibdata = controller.GetBufferData(self.indexResourceId, self.indexByteOffset, 0)

            # 计算字节偏移量
            offset_bytes = self.indexOffset * self.indexByteStride

            # 从字节数据中读取索引数组
            indices = np.frombuffer(ibdata, dtype=dtype, count=self.numIndices, offset=offset_bytes)

            # 转换为大整数类型后加上基址顶点，避免溢出
            indices = indices.astype(np.int64) + self.baseVertex

            return np.ascontiguousarray(indices)
        else:
            raise Exception

    def fetchData(self, controller, indices=None):
        if indices is None:
            indices = self.fetchIndices(controller)
        if len(indices) == 0:
            return np.empty(0, dtype=np.float32)

        # 获取原始二进制数据
        raw_data = controller.GetBufferData(self.vertexResourceId, self.vertexByteOffset, 0)

        # 计算关键参数
        data_per_vertex = struct.calcsize(self.format_str)
        vertex_stride = self.vertexByteStride
        num_vertices = (len(raw_data) - data_per_vertex) // vertex_stride + 1

        # 类型映射
        type_char = self.format_str[-1]
        np_type = MeshData.TYPE_MAP.get(type_char, np.uint8)
        comp_count = self.format.compCount

        # 优化分支：当无填充数据时直接转换
        if data_per_vertex == vertex_stride:
            # 直接转换整个缓冲区
            value_array = np.frombuffer(raw_data, dtype=np_type)
            value_array = value_array.reshape(-1, comp_count)[:num_vertices]  # 确保形状正确
        else:
            # 有填充数据时需要跨步视图
            sliced_bytes = np.lib.stride_tricks.as_strided(
                np.frombuffer(raw_data, dtype=np.uint8),
                shape=(num_vertices, data_per_vertex),
                strides=(vertex_stride, 1)
            )
            value_array = np.frombuffer(sliced_bytes.tobytes(), dtype=np_type)
            value_array = value_array.reshape(num_vertices, comp_count)

        # ---- 后续统一的后处理流程保持不变 ----
        # 向量化归一化处理
        if self.format.compType == rd.CompType.UNorm:
            divisor = (1 << (self.format.compByteWidth * 8)) - 1
            value_array = value_array.astype(np.float32) / divisor
        elif self.format.compType == rd.CompType.SNorm:
            max_neg = -(1 << (self.format.compByteWidth * 8 - 1))
            divisor = -(max_neg - 1)
            value_array = np.where(
                value_array == max_neg,
                np.float32(max_neg),
                value_array.astype(np.float32) / divisor
            )

        # BGRA顺序调整
        if self.format.BGRAOrder() and value_array.shape[1] == 4:
            value_array = value_array[:, [2, 1, 0, 3]]
        if not value_array.flags.writeable:
            value_array = np.copy(np.ascontiguousarray(value_array))
        return value_array


class DrawData:
    def __init__(self, vertices, triangles, uvs, texture, constants):
        self.vertices = vertices
        self.triangles = triangles
        self.uvs = uvs
        self.texture = texture
        self.constants = constants


class BatchData:
    def __init__(self, file_name=""):
        self.file_name = file_name
        self.draw_datas: dict[int, DrawData] = {}

    def addDrawData(self, drawcallId: int, data: DrawData):
        self.draw_datas[drawcallId] = data


class CaptureScraper:
    def __init__(self, file_name, controller, max_blocks=-1, global_scale=1.0 / 256.0, ref_matrix=None, **kwargs):
        self.file_name = file_name
        self.controller = controller
        self.max_blocks = max_blocks
        self.global_scale = global_scale
        self.ref_matrix = ref_matrix
        self.debug_matrix = kwargs.get("debug_matrix", False)
        self.debug_color = kwargs.get("debug_color", None)

        all_texture_desc: list[rd.TextureDescription] = self.controller.GetTextures()
        self._rid2desc = {desc.resourceId: desc for desc in all_texture_desc}
        self._cached_constants = {}

    def _findDrawcallBatch(self, drawcalls, first_call_prefix, drawcall_prefix, last_call_prefix):
        batch = []
        has_batch_started = False
        last_call_index = 0
        for last_call_index, draw in enumerate(drawcalls):
            if draw.name.startswith(first_call_prefix) and not has_batch_started:
                has_batch_started = True
            if not has_batch_started:
                continue
            # ↓↓ has batch stated == True ↓↓
            if (not draw.name.startswith(drawcall_prefix)) and draw.name.startswith(last_call_prefix) and len(batch) > 0:
                break
            if draw.name.startswith(drawcall_prefix):
                batch.append(draw)

        return batch, last_call_index

    def _getVertexShaderConstants(self, draw, state=None):
        if draw.eventId in self._cached_constants:
            return self._cached_constants[draw.eventId]
        if state is None:
            self.controller.SetFrameEvent(draw.eventId, True)
            state = self.controller.GetPipelineState()

        shader = state.GetShader(rd.ShaderStage.Vertex)
        ep = state.GetShaderEntryPoint(rd.ShaderStage.Vertex)
        ref = state.GetShaderReflection(rd.ShaderStage.Vertex)
        constants = {}
        for cbn, cb in enumerate(ref.constantBlocks):
            block = {}
            cbuff = state.GetConstantBuffer(rd.ShaderStage.Vertex, cbn, 0)
            variables = self.controller.GetCBufferVariableContents(
                state.GetGraphicsPipelineObject(),
                shader,
                ep,
                cb.bindPoint,
                cbuff.resourceId,
                0,
                0
            )
            for var in variables:
                val = 0
                if var.members:
                    val = []
                    for member in var.members:
                        member_val = 0
                        if member.type == rd.VarType.Float:
                            member_val = member.value.f32v[:member.rows * member.columns]
                        elif member.type == rd.VarType.Int:
                            member_val = member.value.s32v[:member.rows * member.columns]
                        else:
                            logging.warning(f"Unsupported type! {member.type}")
                        # ...
                        val.append(member_val)
                else:
                    if var.type == rd.VarType.Float:
                        val = var.value.f32v[:var.rows * var.columns]
                    elif var.type == rd.VarType.Int:
                        val = var.value.s32v[:var.rows * var.columns]
                    else:
                        logging.warning(f"Unsupported type! {var.type}")
                    # ...
                block[var.name] = val
            constants[cb.name] = block
        self._cached_constants[draw.eventId] = constants
        return constants

    def _hasUniform(self, draw, uniform):
        constants = self._getVertexShaderConstants(draw)
        return uniform in constants['$Globals']

    def _extractRelevantCalls(self, drawcalls):
        # With Google Earth there are two batches of DrawIndexed calls, we are interested in the second one
        first_call = "DrawIndexed"
        last_call = ""
        drawcall_prefix = "DrawIndexed"
        min_drawcall = 0
        while True:
            skipped_drawcalls, new_min_drawcall = self._findDrawcallBatch(drawcalls[min_drawcall:], first_call, drawcall_prefix, last_call)
            if not skipped_drawcalls or self._hasUniform(skipped_drawcalls[0], MATRIX_NAME):
                break
            min_drawcall += new_min_drawcall

        relevant_drawcalls, _ = self._findDrawcallBatch(
            drawcalls[min_drawcall:],
            first_call,
            drawcall_prefix,
            last_call)
        if not relevant_drawcalls:
            raise Exception("Couldn't find relevant drawcalls")

        relevant_drawcalls = [call for call in relevant_drawcalls if self._hasUniform(call, MATRIX_NAME)]
        return relevant_drawcalls

    def _extractTexture(self, drawcallId, state, resize=False) -> Optional[np.ndarray]:
        bindpoints = state.GetBindpointMapping(rd.ShaderStage.Fragment)
        if not bindpoints.samplers:
            logging.warning(f"No texture found for drawcall {drawcallId}")
            return None
        texture_bind = bindpoints.samplers[-1].bind
        resources = state.GetReadOnlyResources(rd.ShaderStage.Fragment)
        rid = resources[texture_bind].resources[0].resourceId

        tex_desc: rd.TextureDescription = self._rid2desc[rid]
        tex_format: rd.ResourceFormat = tex_desc.format
        width, height = tex_desc.width, tex_desc.height
        tex_format_str = tex_format.Name()
        if tex_format_str != "BC1_UNORM":
            logging.warning(f"Texture format {tex_format_str} is not supported!")
            return None
        tex_data: bytes = self.controller.GetTextureData(rid, rd.Subresource())
        decoded_data = texture2ddecoder.decode_bc1(tex_data, width, height)
        tex_bgra = np.frombuffer(decoded_data, dtype=np.uint8).reshape((height, width, -1))
        if resize:
            new_width, new_height = width // 2, height // 2
            tex_bgra = cv2.resize(
                tex_bgra,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC  # 或使用INTER_AREA保持锐利
            )

        tex_rgba = np.ascontiguousarray(tex_bgra[..., [2, 1, 0, 3]])  # BGRA to RGBA
        if self.debug_color is not None:
            tex_rgba = (np.array(self.debug_color, dtype=np.float32) * tex_rgba.astype(np.float32)).astype(np.uint8)
        return tex_rgba

    def _extractUniforms(self, constants, refMatrix):
        """Extract from constant buffer the model matrix and uv offset
        The reference matrix is used to cancel the view part of teh modelview matrix
        """
        _ = self
        # Extract constants, which have different names depending on the browser/GPU driver
        globUniforms = constants['$Globals']
        assert MATRIX_NAME in globUniforms, "We Only Support Google Earth"
        # Google Earth
        # dict_keys(['_uProjectionMatrix',
        # '_uModelviewMatrix',
        # '_uWorldOriginInEye',
        # '_uDrapedProjModelviewMatrix',
        # '_uDrapedEye',
        # '_uDrapedTextureInsetOffsetScale',
        # '_uMeshToWorldMatrix',
        # '_uDrapedDpOffset',
        # '_uNoDraw_NoDrapeLayer',
        # '_uTexCoordScaleBias'])
        uvOffsetScale = [0, -1, 1, -1]
        if not self.debug_matrix:
            matrix: pyrr.Matrix44 = pyrr.Matrix44(globUniforms[MATRIX_NAME], dtype=np.float64).T
            matrix[3] = [0, 0, 0, 1]
            if refMatrix is None:
                refMatrix = matrix.inverse * pyrr.Quaternion.from_y_rotation(-np.pi / 2.0)
            matrix = matrix * refMatrix
        else:
            M = pyrr.Matrix44(globUniforms[MATRIX_NAME], dtype=np.float64).T
            V = pyrr.Matrix44(globUniforms['_uModelviewMatrix'], dtype=np.float64).T
            # P = pyrr.Matrix44(globUniforms['_uProjectionMatrix'], dtype=np.float64).T
            M[3] = [0, 0, 0, 1]
            matrix = M * V
            if refMatrix is None:
                refMatrix = matrix.inverse * pyrr.Quaternion.from_y_rotation(-np.pi / 2.0)
            matrix = matrix * refMatrix
        return uvOffsetScale, matrix, refMatrix

    def run(self, show_progress=False, print_duration=False) -> BatchData:
        start_time = time.time()
        drawcalls = self.controller.GetDrawcalls()
        duration = time.time() - start_time
        if print_duration:
            logging.info(f"Found {len(drawcalls)} drawcalls in {duration:.2f}s")
        start_time = time.time()
        relevant_drawcalls = self._extractRelevantCalls(drawcalls)
        duration = time.time() - start_time
        if print_duration:
            logging.info(f"Found {len(relevant_drawcalls)} relevant drawcalls in {duration:.2f}s")
        if self.max_blocks <= 0:
            max_drawcall = len(relevant_drawcalls)
        else:
            max_drawcall = min(self.max_blocks, len(relevant_drawcalls))
        out_batch_data = BatchData(self.file_name)
        start_time = time.time()
        for drawcallId, draw in tqdm(enumerate(relevant_drawcalls[:max_drawcall]), total=max_drawcall, desc="Extracting Data", disable=not show_progress):
            self.controller.SetFrameEvent(draw.eventId, True)
            state = self.controller.GetPipelineState()
            ib = state.GetIBuffer()
            vbs = state.GetVBuffers()
            attrs = state.GetVertexInputs()
            mesh_datas = [MeshData(attr, ib, vbs, draw) for attr in attrs]

            # Position
            m = mesh_datas[0]
            indices = m.fetchIndices(self.controller)
            vertices = m.fetchData(self.controller, indices=indices)

            # UV
            m = mesh_datas[2]
            uvs = m.fetchData(self.controller)

            # Vertex Shader Constants
            constants = self._getVertexShaderConstants(draw, state=state)
            constants["DrawCall"] = {
                "topology": 'TRIANGLE_STRIP' if state.GetPrimitiveTopology() == rd.Topology.TriangleStrip else 'TRIANGLES',
                "type": "Google Earth"
            }
            # texture
            tex = self._extractTexture(drawcallId, state)

            # ===============后处理===============
            # 处理坐标变换
            uv_offset_scale, matrix, self.ref_matrix = self._extractUniforms(constants, self.ref_matrix)
            triangles = indices.reshape(-1, 3)

            # 处理UV坐标
            ou, ov, su, sv = uv_offset_scale
            uvs = (uvs + np.array([ou, ov])) * np.array([su, sv])

            # 应用矩阵变换
            vertices[:, 3] = 1
            world_matrix = matrix * pyrr.Matrix44.from_scale(pyrr.Vector3([self.global_scale] * 3, dtype="f4"))
            vertices = vertices @ np.array(world_matrix.T)
            vertices = vertices[:, :3]

            # 添加到输出
            out_batch_data.addDrawData(drawcallId, DrawData(vertices, triangles, uvs, tex, constants))
        duration = time.time() - start_time
        if print_duration:
            logging.info(f"Extracted {len(out_batch_data.draw_datas)} drawcalls in {duration:.2f}s")
        return out_batch_data


def extract_data_from_rdc(file_name: str, show_progress=False, print_duration=False, **kwargs) -> BatchData:
    with CaptureWrapper(file_name) as controller:
        cap_scraper = CaptureScraper(file_name, controller, **kwargs)
        extracted_data: BatchData = cap_scraper.run(show_progress, print_duration)
    return extracted_data


def create_coordinate_axis(origin=[0, 0, 0], size=1.0):
    """
    创建表示坐标轴的 LineSet
    参数:
        origin (list): 坐标原点 [x, y, z]
        size (float): 坐标轴长度
    返回:
        o3d.geometry.LineSet: 包含 X/Y/Z 轴的线段集合
    """
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


def create_open3d_meshes(batch_data: BatchData, show_progress=False, print_duration=False) -> list[o3d.geometry.TriangleMesh]:
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



if __name__ == "__main__":
    meshes = []

    file_name = "./results/2025-04-29_22-31-12/rdc/1_1.rdc"
    batch_data = extract_data_from_rdc(file_name, show_progress=False, print_duration=True)
    meshes.extend(create_open3d_meshes(batch_data, show_progress=False, print_duration=True))

    meshes.append(create_coordinate_axis(size=5.0))
    o3d.visualization.draw_geometries(meshes)
