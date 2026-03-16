#!/usr/bin/env python3
"""
Rope Joint Demo
===============

演示基于 MuJoCo 关节系统的绳索仿真。
这是工业级仿真的正道：把绳索当作"最大长度可变、只能受拉不能受压的弹簧"，
把运算资源留给刚体动力学。

控制方式：
    A / D      → 轨道 Y 轴
    W / S      → 小车 X 轴
    Q / E      → 绳索伸长/缩短
    R / T      → 锚点升降
    左键拖动   → 旋转视角
    右键拖动   → 平移视角
    滚轮       → 缩放
    ESC        → 退出
"""

import time
import glfw
import numpy as np
import mujoco

# 导入我们的新绳索仿真类
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rope_sim_Joint.rope_sim_Joint import RopeJointSim


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
        """应用相机状态到 MuJoCo 相机。"""
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation
        cam.distance = self.distance
        cam.lookat[:] = self.lookat

    def on_mouse_button(self, window, button, action, mods):
        self._btn_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._btn_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if action == glfw.RELEASE:
            self._last_x = None
            self._last_y = None

    def on_cursor_pos(self, window, x, y):
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
        self.distance = float(np.clip(self.distance * (1.0 - yoff * 0.1), 1.0, 200.0))


def build_vis_xml(max_rope_segments: int = 50):
    """构建可视化用的 MuJoCo XML。

    这个模型只用于渲染，不进行物理计算。
    """
    rope_segments = []
    for i in range(max_rope_segments):
        rope_segments.append(f'''
        <body name="rope_{i}" mocap="true">
            <geom name="rope_geom_{i}" type="capsule" size="0.03 0.5"
                  rgba="0.8 0.6 0.2 1"/>
        </body>
        ''')

    return f'''
<mujoco model="rope_vis">
    <option timestep="0.002"/>
    <worldbody>
        <light name="light" pos="0 0 20" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="50 50 0.1" rgba="0.8 0.9 0.8 1"/>

        <!-- 起重机静态可视化 -->
        <body name="crane_base" pos="0 0 0">
            <geom name="base" type="box" size="3 3 0.5" rgba="0.5 0.5 0.5 1"/>
            <geom name="pillar" type="box" size="0.5 0.5 6" rgba="0.6 0.6 0.6 1" pos="0 0 6"/>
            <geom name="beam" type="box" size="8 0.5 0.5" rgba="0.6 0.6 0.6 1" pos="0 0 12"/>
        </body>

        <!-- 锚点标记 -->
        <body name="hoist_marker" mocap="true">
            <geom name="hoist_geom" type="box" size="0.5 0.5 0.5" rgba="0.8 0.2 0.2 1"/>
        </body>

        <!-- 末端夹具 -->
        <body name="clamp" mocap="true">
            <geom name="clamp_geom" type="sphere" size="0.35" rgba="0.2 0.5 0.8 1"/>
        </body>

        <!-- 绳索段（用于可视化） -->
        {''.join(rope_segments)}
    </worldbody>
</mujoco>
    '''


def quat_from_z_to_vec(vec):
    """计算从 z 轴到目标向量的四元数。"""
    vec = np.array(vec, dtype=np.float64)
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    vec = vec / vec_norm

    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, vec)
    dot = np.dot(z_axis, vec)

    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])

    q = np.zeros(4)
    q[0] = 1.0 + dot
    q[1:] = cross
    q = q / np.linalg.norm(q)
    return q


def main():
    print("=" * 60)
    print("  Rope Joint Demo - MuJoCo 关节系统绳索仿真")
    print("=" * 60)
    print("  这是工业级仿真的正道：")
    print("  - 把绳索当作'最大长度可变、只能受拉不能受压的弹簧'")
    print("  - 完全依赖 MuJoCo 的刚体动力学引擎")
    print("  - 把运算资源留给重物摆动和机械臂力矩计算")
    print("=" * 60)
    print("  A / D      → 轨道 Y 轴")
    print("  W / S      → 小车 X 轴")
    print("  Q / E      → 绳索伸长/缩短")
    print("  R / T      → 锚点升降")
    print("  左键拖动   → 旋转视角")
    print("  右键拖动   → 平移视角")
    print("  滚轮       → 缩放")
    print("  ESC        → 退出")
    print("=" * 60)

    # 物理引擎参数
    HOIST_HEIGHT = 12.0
    INITIAL_LENGTH = 8.0
    MAX_LENGTH = 14.0
    MIN_LENGTH = 2.0
    MOVE_SPEED = 0.05
    LENGTH_SPEED = 0.08

    # 创建绳索仿真（基于 MuJoCo 关节系统）
    initial_anchor = np.array([0.0, 0.0, HOIST_HEIGHT])
    rope = RopeJointSim(
        anchor=initial_anchor,
        length=INITIAL_LENGTH,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
        end_mass=10.0,  # 10kg 重物
        rope_stiffness=20000.0,
        rope_damping=200.0,
    )

    # 创建可视化模型
    MAX_ROPE_SEGMENTS = 50
    vis_xml = build_vis_xml(MAX_ROPE_SEGMENTS)
    vis_model = mujoco.MjModel.from_xml_string(vis_xml)
    vis_data = mujoco.MjData(vis_model)

    # 获取 mocap 体 ID
    def get_mocap_id(model, name):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mocap_id = -1
        for i in range(model.nmocap):
            if model.body_mocapid[body_id] == i:
                mocap_id = i
                break
        return mocap_id

    hoist_id = get_mocap_id(vis_model, "hoist_marker")
    clamp_id = get_mocap_id(vis_model, "clamp")
    rope_ids = [get_mocap_id(vis_model, f"rope_{i}") for i in range(MAX_ROPE_SEGMENTS)]

    # 初始化 GLFW
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")

    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(1280, 720, "Rope Joint Demo - MuJoCo", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # 初始化 MuJoCo 渲染对象
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(vis_model, maxgeom=10000)
    ctx = mujoco.MjrContext(vis_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    # 相机状态
    cam_state = CameraState()
    glfw.set_mouse_button_callback(window, cam_state.on_mouse_button)
    glfw.set_cursor_pos_callback(window, cam_state.on_cursor_pos)
    glfw.set_scroll_callback(window, cam_state.on_scroll)

    # 起重机状态
    track_y = 0.0
    cabin_x = 0.0
    hoist_z = 0.0
    current_length = INITIAL_LENGTH
    dt = vis_model.opt.timestep

    # 性能统计
    step_count = 0
    last_print_time = time.time()

    while not glfw.window_should_close(window):
        start_time = time.time()

        # 键盘控制
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            track_y += MOVE_SPEED
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            track_y -= MOVE_SPEED
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            cabin_x += MOVE_SPEED
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            cabin_x -= MOVE_SPEED
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            current_length += LENGTH_SPEED
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            current_length -= LENGTH_SPEED
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            hoist_z += MOVE_SPEED
        if glfw.get_key(window, glfw.KEY_T) == glfw.PRESS:
            hoist_z -= MOVE_SPEED
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # 限制范围
        current_length = float(np.clip(current_length, MIN_LENGTH, MAX_LENGTH))
        hoist_z = float(np.clip(hoist_z, 0.0, HOIST_HEIGHT - 1.0))
        anchor = np.array([cabin_x, track_y, HOIST_HEIGHT - hoist_z])

        # 更新物理引擎
        rope.set_anchor(anchor)
        rope.set_length(current_length)
        rope.step(dt)

        # 获取绳索数据
        end_pos = rope.end_point
        current_dist = rope.get_current_distance()
        tendon_force = rope.get_tendon_force()

        # 更新可视化
        vis_data.mocap_pos[hoist_id] = anchor
        vis_data.mocap_quat[hoist_id] = np.array([1, 0, 0, 0])
        vis_data.mocap_pos[clamp_id] = end_pos
        vis_data.mocap_quat[clamp_id] = np.array([1, 0, 0, 0])

        # 更新绳索段可视化
        rope_segments = rope.get_rope_segments(num_segments=MAX_ROPE_SEGMENTS)
        for i in range(MAX_ROPE_SEGMENTS - 1):
            p1 = rope_segments[i]
            p2 = rope_segments[i + 1]
            vis_data.mocap_pos[rope_ids[i]] = (p1 + p2) * 0.5
            vis_data.mocap_quat[rope_ids[i]] = quat_from_z_to_vec(p2 - p1)

        # 运行正向动力学
        mujoco.mj_forward(vis_model, vis_data)

        # 渲染
        glfw.poll_events()
        cam_state.apply(cam)
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
        mujoco.mjv_updateScene(
            vis_model, vis_data, opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_render(viewport, scene, ctx)
        glfw.swap_buffers(window)

        # 性能统计
        step_count += 1
        if time.time() - last_print_time > 2.0:
            print(f"\r  绳索长度: {current_length:.2f}m | "
                  f"实际距离: {current_dist:.2f}m | "
                  f"绳索张力: {tendon_force:.1f}N | "
                  f"末端位置: ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f})",
                  end="", flush=True)
            last_print_time = time.time()

        # 控制帧率
        elapsed = time.time() - start_time
        time.sleep(max(0.0, dt - elapsed))

    print("\n" + "=" * 60)
    print("  仿真结束")
    print("=" * 60)

    ctx.free()
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
