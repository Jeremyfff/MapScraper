
import abc
import logging
import os
import os.path
import struct
import sys
import threading
import time
from collections import defaultdict
from typing import Callable
from typing import Optional

import cv2
import numpy as np
import pyrr
import texture2ddecoder
from tqdm import tqdm

from src.dataclasses import BatchData, DrawData

win64_path = os.path.join(".", "bin", "win64")

# 添加路径到 sys.path
if win64_path not in sys.path:
    sys.path.insert(0, win64_path)
try:
    import renderdoc

    logging.info("renderdoc 模块已加载")
except Exception as e:
    raise Exception(f"renderdoc 模块导入失败, {e}")

rd = renderdoc


class TargetControllerWrapper:
    def __init__(self, driver, controller_name="my_client", ):
        assert driver is not None, "driver is None"  # 为了确保已经注入了driver后再启动该wrapper
        self.targetControl: Optional[rd.TargetControl] = None
        self.controller_name = controller_name

    def __enter__(self):
        """确保进入时，程序已经被注入"""
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
        target_control: rd.TargetControl = rd.CreateTargetControl("", ident, self.controller_name, True)
        assert target_control is not None, "Create Failed"
        assert target_control.Connected()
        logging.info("Target Control Connected")
        self.targetControl = target_control
        self.print_target_control_info()
        return target_control

    def __exit__(self, _type, _value, _traceback):
        self.targetControl.Shutdown()
        logging.info("Target Control Shutdown")

    def print_target_control_info(self):
        if self.targetControl is None:
            logging.warning("Target Control is None")
            return
        logging.info(f"Connected: {self.targetControl.Connected()}")
        logging.info(f"API: {self.targetControl.GetAPI()}")
        logging.info(f"BusyClient: {self.targetControl.GetBusyClient()}")
        logging.info(f"PID: {self.targetControl.GetPID()}")
        logging.info(f"Target: {self.targetControl.GetTarget()}")


def inject_into_process(pid, capture_options: rd.CaptureOptions = None):
    assert pid > 0, "PID must be greater than 0"
    if capture_options is None:
        capture_options = rd.GetDefaultCaptureOptions()
    result = rd.InjectIntoProcess(pid, [], os.path.join(os.path.abspath('.') + "/"), capture_options, False)
    if result.status == rd.ReplayStatus.Succeeded:
        logging.info("Injection Success! Now you can close the PID Popup Window")
    else:
        raise Exception(f"Injection Failed! status: {result.status}")


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

    def __exit__(self, _type, _value, _traceback):
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


