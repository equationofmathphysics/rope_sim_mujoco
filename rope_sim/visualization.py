"""
Rope Visualization
==================

使用 MuJoCo 进行绳索可视化的模块。
完全依赖物理引擎数据，不包含任何物理计算。
"""

import time
import mujoco
import numpy as np
import glfw


class CameraState:
    """相机状态管理类，处理用户交互。"""

    def __init__(self):
        self.azimuth = 45.0
        self.elevation = -20.0
        self.distance = 30.0
        self.lookat = np.array([0.0, 0.0, 5.0])

        self._last_x = None
        self._last_y = None
        self._btn_left = False
        self._btn_right = False

    def apply(self, cam):
        """应用相机状态到 MuJoCo 相机。

        Args:
            cam: MuJoCo MjvCamera 对象
        """
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation
        cam.distance = self.distance
        cam.lookat[:] = self.lookat

    def on_mouse_button(self, window, button, action, mods):
        """鼠标按钮回调。"""
        self._btn_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._btn_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if action == glfw.RELEASE:
            self._last_x = None
            self._last_y = None

    def on_cursor_pos(self, window, x, y):
        """光标移动回调。"""
        if self._last_x is None:
            self._last_x, self._last_y = x, y
            return
        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x, self._last_y = x, y

        if self._btn_left:
            self.azimuth -= dx * 0.4
            self.elevation = float(np.clip(self.elevation + dy * 0.4, -89, 89))
        elif self._btn_right:
            azimuth_rad = np.radians(self.azimuth)
            right = np.array([np.cos(azimuth_rad), -np.sin(azimuth_rad), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            scale = self.distance * 0.001
            self.lookat -= right * dx * scale
            self.lookat += up * dy * scale

    def on_scroll(self, window, xoff, yoff):
        """滚轮回调。"""
        self.distance = float(np.clip(self.distance * (1.0 - yoff * 0.1), 1.0, 200.0))


class RopeVisualizer:
    """绳索可视化器类。

    使用 MuJoCo 渲染绳索物理状态。

    Args:
        rope_physics: RopePhysics 实例
        window_size: 窗口大小 (width, height)
        title: 窗口标题
    """

    def __init__(self, rope_physics, window_size=(1280, 720), title="Rope Simulation"):
        self.rope_physics = rope_physics
        self.window_size = window_size
        self.title = title

        # 创建 MuJoCo 模型 - 必须使用 max_segments 以支持伸长
        xml = build_rope_xml(rope_physics.max_segments)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # 获取 mocap 体 ID
        self.hoist_id = get_mocap_id(self.model, "hoist_marker")
        self.clamp_id = get_mocap_id(self.model, "clamp")
        self.rope_ids = [get_mocap_id(self.model, f"rope_{i}") for i in range(rope_physics.max_segments - 1)]

        # 初始化 GLFW
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(window_size[0], window_size[1], title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # 初始化 MuJoCo 渲染对象
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)

        # 相机状态
        self.cam_state = CameraState()
        glfw.set_mouse_button_callback(self.window, self.cam_state.on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.cam_state.on_cursor_pos)
        glfw.set_scroll_callback(self.window, self.cam_state.on_scroll)

        self.dt = float(self.model.opt.timestep)

    def update(self, anchor):
        """更新可视化状态。

        Args:
            anchor: 绳索锚点位置 (x, y, z)
        """
        # 更新 mocap 位置
        self.data.mocap_pos[self.hoist_id] = anchor
        self.data.mocap_quat[self.hoist_id] = np.array([1, 0, 0, 0])

        # 更新绳索可视化 - 只更新 active segments
        positions = self.rope_physics.positions
        for i in range(self.rope_physics.active_segments - 1):
            p1 = positions[i]
            p2 = positions[i + 1]
            self.data.mocap_pos[self.rope_ids[i]] = (p1 + p2) * 0.5
            self.data.mocap_quat[self.rope_ids[i]] = quat_from_z_to_vec(p2 - p1)

        # 隐藏未 active 的 segments（设置到很远的地方）
        for i in range(self.rope_physics.active_segments - 1, self.rope_physics.max_segments - 1):
            self.data.mocap_pos[self.rope_ids[i]] = np.array([0, 0, -1000])

        # 更新末端夹具
        self.data.mocap_pos[self.clamp_id] = self.rope_physics.end_point
        self.data.mocap_quat[self.clamp_id] = np.array([1, 0, 0, 0])

        # 运行 MuJoCo 正向动力学（渲染需要）
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        """渲染一帧。"""
        glfw.poll_events()

        # 应用相机状态
        self.cam_state.apply(self.cam)

        # 渲染
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.ctx)

        glfw.swap_buffers(self.window)

    def is_running(self):
        """检查是否需要继续运行。

        Returns:
            True 如果窗口未关闭
        """
        return not glfw.window_should_close(self.window)

    def shutdown(self):
        """关闭可视化器。"""
        if hasattr(self, "ctx"):
            self.ctx.free()
        if hasattr(self, "window"):
            glfw.destroy_window(self.window)
        glfw.terminate()

    def get_key_state(self, key):
        """获取键盘按键状态。

        Args:
            key: GLFW 按键常量

        Returns:
            True 如果按键被按下
        """
        return glfw.get_key(self.window, key) == glfw.PRESS


# 导入 utils 函数（为了方便使用）
from .utils import quat_from_z_to_vec, build_rope_xml, get_mocap_id
