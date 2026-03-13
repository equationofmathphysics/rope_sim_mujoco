
"""
Crane Rope Demo
================
架构:
    Python  → crane 运动学 + rope physics (PBD)
    MuJoCo  → 纯渲染 (mocap bodies, 直接 GLFW + MjrContext)

控制:
    A / D       → 轨道 Y 轴
    W / S       → 小车 X 轴
    Q / E       → 升降
    鼠标左键拖动 → 旋转视角
    鼠标右键拖动 → 平移视角
    滚轮         → 缩放
    ESC          → 退出
"""

import mujoco
import numpy as np
import glfw
import time


# ─────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────

ROPE_LENGTH  = 10.0
ROPE_SEG_LEN = 0.1
ROPE_RADIUS  = 0.02
ROPE_COLOR   = "0.15 0.15 0.15 1"

HOIST_HEIGHT = 12.0
MOVE_SPEED   = 0.05
PBD_ITERS    = 8

WIN_W, WIN_H = 1280, 720


# ─────────────────────────────────────────────────────────
# Rope Physics (PBD + Verlet)
# ─────────────────────────────────────────────────────────

class RopePhysics:

    def __init__(self, anchor, length=ROPE_LENGTH, seg_len=ROPE_SEG_LEN):
        self.seg_len = seg_len
        self.n       = int(round(length / seg_len))
        self.gravity = np.array([0.0, 0.0, -9.81])

        self.pos  = np.array([anchor + np.array([0, 0, -i * seg_len])
                               for i in range(self.n)], dtype=float)
        self.prev = self.pos.copy()

    def set_anchor(self, anchor):
        self.pos[0]  = anchor
        self.prev[0] = anchor

    def step(self, dt):
        for i in range(1, self.n):
            vel          = self.pos[i] - self.prev[i]
            self.prev[i] = self.pos[i].copy()
            self.pos[i] += vel + self.gravity * dt * dt

        for _ in range(PBD_ITERS):
            for i in range(self.n - 1):
                d    = self.pos[i+1] - self.pos[i]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    continue
                corr = d * ((dist - self.seg_len) / dist)
                if i == 0:
                    self.pos[i+1] -= corr
                else:
                    self.pos[i]   += corr * 0.5
                    self.pos[i+1] -= corr * 0.5

    @property
    def end_point(self):
        return self.pos[-1].copy()


# ─────────────────────────────────────────────────────────
# 四元数: +Z 对齐到向量
# ─────────────────────────────────────────────────────────

def quat_from_z_to_vec(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v   = v / norm
    z   = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if dot >  1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis  = np.cross(z, v)
    axis /= np.linalg.norm(axis)
    half  = np.arccos(dot) / 2.0
    s     = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])


# ─────────────────────────────────────────────────────────
# XML
# ─────────────────────────────────────────────────────────

def build_xml(n_segments):
    rope_bodies = ""
    for i in range(n_segments - 1):
        rope_bodies += f"""
        <body name="rope_{i}" mocap="true">
            <geom type="capsule"
                  size="{ROPE_RADIUS} {ROPE_SEG_LEN * 0.5}"
                  rgba="{ROPE_COLOR}"
                  contype="0" conaffinity="0"/>
        </body>"""

    return f"""
<mujoco model="crane_demo">

    <option timestep="0.01" gravity="0 0 -9.81"/>

    <visual>
        <map znear="0.05" zfar="500"/>
        <global offwidth="{WIN_W}" offheight="{WIN_H}"/>
    </visual>

    <worldbody>

        <light name="sun" pos="0 0 40" dir="0 0 -1"
               diffuse="1 1 1" ambient="0.4 0.4 0.4" castshadow="false"/>

        <geom type="plane" size="100 100 0.1"
              rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>

        <body name="hoist_marker" mocap="true">
            <geom type="sphere" size="0.12"
                  rgba="1 0.8 0 1" contype="0" conaffinity="0"/>
        </body>

        <body name="clamp" mocap="true">
            <geom type="sphere" size="0.25"
                  rgba="0.9 0.2 0.2 1" contype="0" conaffinity="0"/>
        </body>

        <body name="axes">
            <geom type="cylinder" pos="3 0 0.01" size="0.03 3"
                  euler="0 90 0" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
            <geom type="cylinder" pos="0 3 0.01" size="0.03 3"
                  euler="90 0 0" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
            <geom type="cylinder" pos="0 0 3" size="0.03 3"
                  rgba="0 0 1 0.5" contype="0" conaffinity="0"/>
        </body>

        {rope_bodies}

    </worldbody>

</mujoco>
"""


# ─────────────────────────────────────────────────────────
# mocap 索引
# ─────────────────────────────────────────────────────────

def get_mocap_id(model, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return model.body_mocapid[body_id]


# ─────────────────────────────────────────────────────────
# 更新 mocap visuals
# ─────────────────────────────────────────────────────────

def update_visuals(data, rope, rope_ids, clamp_id, hoist_id, anchor):
    data.mocap_pos[hoist_id]  = anchor
    data.mocap_quat[hoist_id] = np.array([1, 0, 0, 0])

    for i in range(rope.n - 1):
        p1  = rope.pos[i]
        p2  = rope.pos[i+1]
        data.mocap_pos[rope_ids[i]]  = (p1 + p2) * 0.5
        data.mocap_quat[rope_ids[i]] = quat_from_z_to_vec(p2 - p1)

    data.mocap_pos[clamp_id]  = rope.end_point
    data.mocap_quat[clamp_id] = np.array([1, 0, 0, 0])


# ─────────────────────────────────────────────────────────
# Camera 交互
# ─────────────────────────────────────────────────────────

class CameraState:

    def __init__(self):
        self.azimuth   =  45.0
        self.elevation = -20.0
        self.distance  =  30.0
        self.lookat    = np.array([0.0, 0.0, 5.0])

        self._last_x    = None
        self._last_y    = None
        self._btn_left  = False
        self._btn_right = False

    def apply(self, cam):
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth   = self.azimuth
        cam.elevation = self.elevation
        cam.distance  = self.distance
        cam.lookat[:] = self.lookat

    def on_mouse_button(self, window, button, action, mods):
        self._btn_left  = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)  == glfw.PRESS
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
            self.azimuth   -= dx * 0.4
            self.elevation  = float(np.clip(self.elevation + dy * 0.4, -89, 89))

        elif self._btn_right:
            az    = np.radians(self.azimuth)
            right = np.array([ np.cos(az), -np.sin(az), 0.0])
            up    = np.array([0.0, 0.0, 1.0])
            scale = self.distance * 0.001
            self.lookat -= right * dx * scale
            self.lookat += up    * dy * scale

    def on_scroll(self, window, xoff, yoff):
        self.distance = float(np.clip(self.distance * (1.0 - yoff * 0.1), 1.0, 200.0))


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():

    print("=" * 50)
    print("  Crane Rope Demo")
    print("=" * 50)
    print("  A / D      → 轨道 Y 轴")
    print("  W / S      → 小车 X 轴")
    print("  Q / E      → 升降")
    print("  左键拖动   → 旋转视角")
    print("  右键拖动   → 平移视角")
    print("  滚轮       → 缩放")
    print("  ESC        → 退出")
    print("=" * 50)

    # ── rope ──
    init_anchor = np.array([0.0, 0.0, HOIST_HEIGHT])
    rope        = RopePhysics(anchor=init_anchor)

    # ── MuJoCo ──
    xml      = build_xml(rope.n)
    xml_path = "/tmp/crane_demo.xml"
    with open(xml_path, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    hoist_id = get_mocap_id(model, "hoist_marker")
    clamp_id = get_mocap_id(model, "clamp")
    rope_ids = [get_mocap_id(model, f"rope_{i}") for i in range(rope.n - 1)]

    # ── GLFW ──
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(WIN_W, WIN_H, "Crane Rope Demo", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # ── MuJoCo 渲染对象 ──
    # 用 MjrContext 直接渲染到 GLFW 窗口，不走离屏 Renderer
    cam   = mujoco.MjvCamera()
    opt   = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    ctx   = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    cam_state = CameraState()
    glfw.set_mouse_button_callback(window, cam_state.on_mouse_button)
    glfw.set_cursor_pos_callback(window,   cam_state.on_cursor_pos)
    glfw.set_scroll_callback(window,       cam_state.on_scroll)

    # ── crane 状态 ──
    track_y = 0.0
    cabin_x = 0.0
    hoist_z = 0.0
    dt      = float(model.opt.timestep)

    while not glfw.window_should_close(window):

        t0 = time.time()

        glfw.poll_events()

        # 键盘
        def key(k): return glfw.get_key(window, k) == glfw.PRESS

        if key(glfw.KEY_A):      track_y += MOVE_SPEED
        if key(glfw.KEY_D):      track_y -= MOVE_SPEED
        if key(glfw.KEY_W):      cabin_x += MOVE_SPEED
        if key(glfw.KEY_S):      cabin_x -= MOVE_SPEED
        if key(glfw.KEY_Q):      hoist_z += MOVE_SPEED
        if key(glfw.KEY_E):      hoist_z -= MOVE_SPEED
        if key(glfw.KEY_ESCAPE): break

        hoist_z = float(np.clip(hoist_z, 0.0, HOIST_HEIGHT - 1.0))
        anchor  = np.array([cabin_x, track_y, HOIST_HEIGHT - hoist_z])

        # physics
        rope.set_anchor(anchor)
        rope.step(dt)

        # mocap
        update_visuals(data, rope, rope_ids, clamp_id, hoist_id, anchor)
        mujoco.mj_forward(model, data)

        # 渲染
        cam_state.apply(cam)

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)

        mujoco.mjv_updateScene(
            model, data, opt,
            None, cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene
        )
        mujoco.mjr_render(viewport, scene, ctx)

        glfw.swap_buffers(window)

        elapsed = time.time() - t0
        time.sleep(max(0.0, dt - elapsed))

    ctx.free()
    glfw.terminate()


if __name__ == "__main__":
    main()
