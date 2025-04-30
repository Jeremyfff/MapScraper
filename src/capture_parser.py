import abc
import logging
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import pyrr
import texture2ddecoder
from tqdm import tqdm

from src.dataclasses import BatchData, DrawData
from src.rd_utils import rd, MeshData


class BaseCaptureParser:
    def __init__(self, file_name, controller, max_drawcalls=-1, global_scale=1.0 / 256.0, ref_matrix=None, **kwargs):
        self.file_name = file_name
        self.controller = controller
        self.max_drawcalls = max_drawcalls
        self.global_scale = global_scale
        self.ref_matrix = ref_matrix
        self._debug_matrix = kwargs.get("debug_matrix", False)
        self._debug_color = kwargs.get("debug_color", None)

        all_texture_desc: list[rd.TextureDescription] = self.controller.GetTextures()
        self._rid2desc = {desc.resourceId: desc for desc in all_texture_desc}
        self._cached_constants = defaultdict(dict)  # {rd.ShaderStage: {constants}}

    def _findDrawcallBatch(self, drawcalls, first_call_prefix, drawcall_prefix, last_call_prefix):
        _ = self
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

    @staticmethod
    def _shaderVariable2Py(var: rd.ShaderVariable) -> any:
        """
        rd.VarType contains ['Bool', 'ConstantBlock', 'Double', 'Float', 'GPUPointer', 'Half', 'ReadOnlyResource', 'ReadWriteResource', 'SByte', 'SInt', 'SLong', 'SShort', 'Sampler', 'UByte', 'UInt', 'ULong', 'UShort', 'Unknown']
        see renderdoc.ShaderValue class for more information
        """
        if var.members:
            output = []
            for var_member in var.members:
                output.append(BaseCaptureParser._shaderVariable2Py(var_member))
            return output
        else:
            if var.type == rd.VarType.Float:
                val = var.value.f32v[:var.rows * var.columns]
            elif var.type == rd.VarType.SInt:
                val = var.value.s32v[:var.rows * var.columns]
            elif var.type == rd.VarType.Double:
                val = var.value.f64v[:var.rows * var.columns]
            elif var.type == rd.VarType.UInt:
                val = var.value.u32v[:var.rows * var.columns]
            elif var.type == rd.VarType.Bool:
                val = [bool(v) for v in var.value.u8v[:var.rows * var.columns]] if var.rows * var.columns > 1 else bool(var.value.u8v[0])
            else:
                logging.warning(f"Unsupported type! {var.type}")
                val = None
            return val

    def _getShaderConstants(self, draw, state=None, shader_stage=rd.ShaderStage.Vertex) -> dict:
        if draw.eventId in self._cached_constants[shader_stage]:
            return self._cached_constants[shader_stage][draw.eventId]
        if state is None:
            self.controller.SetFrameEvent(draw.eventId, True)
            state = self.controller.GetPipelineState()

        shader = state.GetShader(shader_stage)
        ep = state.GetShaderEntryPoint(shader_stage)
        ref = state.GetShaderReflection(shader_stage)
        constants = {}
        for cbn, cb in enumerate(ref.constantBlocks):
            block = {}
            cbuff = state.GetConstantBuffer(shader_stage, cbn, 0)
            variables = self.controller.GetCBufferVariableContents(state.GetGraphicsPipelineObject(), shader, ep, cb.bindPoint, cbuff.resourceId, 0, 0)
            for var in variables:
                val = BaseCaptureParser._shaderVariable2Py(var)
                block[var.name] = val
            constants[cb.name] = block
        self._cached_constants[shader_stage][draw.eventId] = constants
        return constants

    def _hasUniform(self, draw, uniform, shader_stage=rd.ShaderStage.Vertex):
        constants = self._getShaderConstants(draw, shader_stage=shader_stage)
        return '$Globals' in constants and uniform in constants['$Globals']

    def _get_mesh_datas(self, draw, state) -> list[MeshData]:
        ib = state.GetIBuffer()
        vbs = state.GetVBuffers()
        attrs = state.GetVertexInputs()
        mesh_datas = [MeshData(attr, ib, vbs, draw) for attr in attrs]
        return mesh_datas

    def _extractTexture(self, drawcallId, state, sampler_index=-1, resize=False) -> Optional[np.ndarray]:
        bindpoints = state.GetBindpointMapping(rd.ShaderStage.Fragment)
        if not bindpoints.samplers:
            logging.warning(f"No texture found for drawcall {drawcallId}")
            return None
        texture_bind = bindpoints.samplers[sampler_index].bind
        resources = state.GetReadOnlyResources(rd.ShaderStage.Fragment)
        rid = resources[texture_bind].resources[0].resourceId

        tex_desc: rd.TextureDescription = self._rid2desc[rid]
        tex_format: rd.ResourceFormat = tex_desc.format
        width, height = tex_desc.width, tex_desc.height
        tex_format_str = tex_format.Name()
        tex_data: bytes = self.controller.GetTextureData(rid, rd.Subresource())
        if tex_format_str == "BC1_UNORM":
            tex_data = texture2ddecoder.decode_bc1(tex_data, width, height)
        elif tex_format_str == "R8G8B8A8_UNORM":
            tex_data = tex_data
        else:
            logging.warning(f"Texture format {tex_format_str} is not supported!")
            return None

        tex_bgra = np.frombuffer(tex_data, dtype=np.uint8).reshape((height, width, -1))
        if resize:
            new_width, new_height = width // 2, height // 2
            tex_bgra = cv2.resize(
                tex_bgra,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC  # 或使用INTER_AREA保持锐利
            )

        tex_rgba = np.ascontiguousarray(tex_bgra[..., [2, 1, 0, 3]])  # BGRA to RGBA
        if self._debug_color is not None:
            tex_rgba = (np.array(self._debug_color, dtype=np.float32) * tex_rgba.astype(np.float32)).astype(np.uint8)
        return tex_rgba

    @abc.abstractmethod
    def stage1_get_relevant_drawcalls(self, drawcalls: list[rd.DrawcallDescription]) -> list[rd.DrawcallDescription]:
        raise NotImplementedError

    @abc.abstractmethod
    def stage2_get_draw_data(self, drawcallId, draw, state, mesh_datas, constants) -> Optional[DrawData]:
        raise NotImplementedError

    def run(self, show_progress=False, print_duration=False) -> BatchData:
        drawcalls = self.controller.GetDrawcalls()
        # Stage1 : Get Relevant Drawcalls
        start_time = time.time()
        relevant_drawcalls = self.stage1_get_relevant_drawcalls(drawcalls)
        duration = time.time() - start_time
        if print_duration:
            logging.info(f"Found {len(relevant_drawcalls)} relevant drawcalls in {duration:.2f}s")
        if self.max_drawcalls <= 0:
            max_drawcall = len(relevant_drawcalls)
        else:
            max_drawcall = min(self.max_drawcalls, len(relevant_drawcalls))

        out_batch_data = BatchData(self.file_name)
        start_time = time.time()
        for drawcallId, draw in tqdm(enumerate(relevant_drawcalls[:max_drawcall]), total=max_drawcall, desc="Extracting Data", disable=not show_progress):
            self.controller.SetFrameEvent(draw.eventId, True)
            state = self.controller.GetPipelineState()
            mesh_datas = self._get_mesh_datas(draw, state)
            # Vertex Shader Constants
            constants = self._getShaderConstants(draw, state=state, shader_stage=rd.ShaderStage.Vertex)
            constants["DrawCall"] = {"topology": 'TRIANGLE_STRIP' if state.GetPrimitiveTopology() == rd.Topology.TriangleStrip else 'TRIANGLES', }

            draw_data = self.stage2_get_draw_data(drawcallId, draw, state, mesh_datas, constants)
            if draw_data is not None:
                out_batch_data.addDrawData(drawcallId, draw_data)
        duration = time.time() - start_time
        if print_duration:
            logging.info(f"Extracted {len(out_batch_data.draw_datas)} drawcalls in {duration:.2f}s")
        return out_batch_data


class GoogleEarthCapturerParser(BaseCaptureParser):
    def _extractUniforms(self, constants):
        """Extract from constant buffer the model matrix and uv offset
        The reference matrix is used to cancel the view part of teh modelview matrix
        """
        # Extract constants, which have different names depending on the browser/GPU driver
        globUniforms = constants['$Globals']
        assert '_uMeshToWorldMatrix' in globUniforms, "We Only Support Google Earth"
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
        matrix: pyrr.Matrix44 = pyrr.Matrix44(globUniforms['_uMeshToWorldMatrix'], dtype=np.float64).T
        matrix[3] = [0, 0, 0, 1]
        if self.ref_matrix is None:
            self.ref_matrix = matrix.inverse * pyrr.Quaternion.from_y_rotation(-np.pi / 2.0)
        matrix = matrix * self.ref_matrix

        return uvOffsetScale, matrix

    def stage1_get_relevant_drawcalls(self, drawcalls: list[rd.DrawcallDescription]) -> list[rd.DrawcallDescription]:
        # With Google Earth there are two batches of DrawIndexed calls, we are interested in the second one
        first_call = "DrawIndexed"
        last_call = ""
        drawcall_prefix = "DrawIndexed"
        min_drawcall = 0
        while True:
            skipped_drawcalls, new_min_drawcall = self._findDrawcallBatch(drawcalls[min_drawcall:], first_call, drawcall_prefix, last_call)
            if not skipped_drawcalls or self._hasUniform(skipped_drawcalls[0], '_uMeshToWorldMatrix'):
                break
            min_drawcall += new_min_drawcall

        relevant_drawcalls, _ = self._findDrawcallBatch(
            drawcalls[min_drawcall:],
            first_call,
            drawcall_prefix,
            last_call)
        if not relevant_drawcalls:
            raise Exception("Couldn't find relevant drawcalls")

        relevant_drawcalls = [call for call in relevant_drawcalls if self._hasUniform(call, '_uMeshToWorldMatrix')]
        return relevant_drawcalls

    def stage2_get_draw_data(self, drawcallId, draw, state, mesh_datas, constants) -> Optional[DrawData]:
        # Position
        m = mesh_datas[0]
        indices = m.fetchIndices(self.controller)
        vertices = m.fetchData(self.controller, indices=indices)

        # UV
        m = mesh_datas[2]
        uvs = m.fetchData(self.controller)

        # texture
        tex = self._extractTexture(drawcallId, state)

        # ===============后处理===============
        # 处理坐标变换
        uv_offset_scale, matrix = self._extractUniforms(constants)
        triangles = indices.reshape(-1, 3)

        # 处理UV坐标
        ou, ov, su, sv = uv_offset_scale
        uvs = (uvs + np.array([ou, ov])) * np.array([su, sv])

        # 应用矩阵变换
        vertices[:, 3] = 1
        world_matrix = matrix * pyrr.Matrix44.from_scale(pyrr.Vector3([self.global_scale] * 3, dtype="f4"))
        vertices = vertices @ np.array(world_matrix.T)
        vertices = vertices[:, :3]
        draw_data = DrawData(vertices=vertices, triangles=triangles, uvs=uvs, texture=tex, constants=constants)
        return draw_data


class BaiduMapCapturerParser(BaseCaptureParser):

    def stage1_get_relevant_drawcalls(self, drawcalls: list[rd.DrawcallDescription]) -> list[rd.DrawcallDescription]:
        relevant_drawcalls, _ = self._findDrawcallBatch(drawcalls, "DrawIndexed", "DrawIndexed", "")
        if not relevant_drawcalls:
            raise Exception("Couldn't find relevant drawcalls")

        # get building draws
        relevant_drawcalls = [call for call in relevant_drawcalls if self._hasUniform(call, '_u_side_light_dir')]
        return relevant_drawcalls

    def stage2_get_draw_data(self, drawcallId, draw, state, mesh_datas, constants) -> Optional[DrawData]:
        # Position
        m = mesh_datas[0]
        indices = m.fetchIndices(self.controller)
        vertices = m.fetchData(self.controller, indices=indices)

        # normal
        # m = mesh_datas[1]
        # normals = m.fetchData(self.controller)

        # color
        m = mesh_datas[2]
        colors = m.fetchData(self.controller)

        # ===============后处理===============
        triangles = indices.reshape(-1, 3)

        # 处理坐标变换
        matrix = self._extractUniforms(constants)
        if vertices.shape[1] == 4:
            vertices[:, 3] = 1
        elif vertices.shape[1] == 3:
            vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        matrix = matrix * pyrr.Matrix44.from_scale(pyrr.Vector3([self.global_scale] * 3, dtype="f4"))
        vertices = vertices @ np.array(matrix.T)
        vertices = vertices[:, :3]
        draw_data = DrawData(vertices=vertices, triangles=triangles, vertex_colors=colors, constants=constants)
        return draw_data

    def _extractUniforms(self, constants):

        _ = self
        # Extract constants, which have different names depending on the browser/GPU driver
        globUniforms = constants['$Globals']
        assert '_u_mv_matrix' in globUniforms

        matrix: pyrr.Matrix44 = pyrr.Matrix44(globUniforms['_u_mv_matrix'], dtype=np.float64).T
        # matrix[3] = [0, 0, 0, 1]
        # if self.ref_matrix is None:
        #     self.ref_matrix = matrix.inverse
        # matrix = matrix * self.ref_matrix

        return matrix
