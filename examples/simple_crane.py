
#!/usr/bin/env python3
"""
球形倒立摆仿真（Spherical Cart-Pole）
======================================
结构：小车（XY 平面平移）--球形铰链-- 倒立杆
物理：MuJoCo 原生刚体动力学
      - 小车：2 个 slide joint（X、Y 轴）
      - 杆：  2 个 hinge joint（绕 X 轴、绕 Y 轴），均无限制
控制：WASD 手动施力 | P 键切换 LQR 自动平衡
"""

import time
import numpy as np
import glfw
import mujoco

# ================================================================
# MuJoCo XML
# ================================================================
XML = """
<mujoco model="spherical_cartpole">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <default>
    <joint damping="0.01" limited="false"/>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>
    <light pos="4 -4 6" dir="-1 1 -1" diffuse="0.4 0.4 0.4"/>
    <geom type="plane" size="25 25 0.1" rgba="0.82 0.82 0.82 1"
          contype="1" conaffinity="1"/>

    <!-- XY 轨道装饰 -->
    <geom type="box" size="12 0.04 0.03" pos="0 0 1.44"
          rgba="0.55 0.55 0.55 1"/>
    <geom type="box" size="0.04 12 0.03" pos="0 0 1.44"
          rgba="0.55 0.55 0.55 1"/>

    <!-- 小车底座：先沿 X 滑动 -->
    <body name="cart_x" pos="0 0 1.44">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="2.0"/>
      <!-- 不可见质量体，满足 MuJoCo 要求 -->
      <geom type="box" size="0.001 0.001 0.001" rgba="0 0 0 0" mass="0.001"/>

      <!-- 再沿 Y 滑动 -->
      <body name="cart_y" pos="0 0 0">
        <joint name="slide_y" type="slide" axis="0 1 0" damping="2.0"/>
        <!-- 不可见质量体，满足 MuJoCo 要求 -->
        <geom type="box" size="0.001 0.001 0.001" rgba="0 0 0 0" mass="0.001"/>
        <geom type="box" size="0.32 0.32 0.14"
              rgba="0.25 0.65 0.40 1" mass="200.0"/>

        <!-- 杆根部铰链：先绕 Y 轴（对应 X 方向倾斜） -->
        <body name="hinge_y" pos="0 0 0.14">
          <joint name="hinge_y" type="hinge" axis="0 1 0" damping="0.008"/>
          <!-- 不可见质量体，满足 MuJoCo 要求 -->
          <geom type="box" size="0.001 0.001 0.001" rgba="0 0 0 0" mass="0.001"/>

          <!-- 再绕 X 轴（对应 Y 方向倾斜） -->
          <body name="hinge_x" pos="0 0 0">
            <joint name="hinge_x" type="hinge" axis="1 0 0" damping="0.008"/>
            <!-- 不可见质量体，满足 MuJoCo 要求 -->
            <geom type="box" size="0.001 0.001 0.001" rgba="0 0 0 0" mass="0.001"/>

            <!-- 杆本体 -->
            <geom type="capsule" fromto="0 0 0 0 0 1.4"
                  size="0.045" rgba="0.80 0.38 0.18 1" mass="0.002"/>
            <!-- 顶端小球 -->
            <geom type="sphere" pos="0 0 1.4" size="0.09"
                  rgba="0.95 0.82 0.10 1" mass="1"/>
          </body>
        </body>

      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="force_x" joint="slide_x" gear="1"
           ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="force_y" joint="slide_y" gear="1"
           ctrllimited="true" ctrlrange="-40 40"/>
  </actuator>

  <sensor>
    <jointpos name="s_x"   joint="slide_x"/>
    <jointvel name="s_xd"  joint="slide_x"/>
    <jointpos name="s_y"   joint="slide_y"/>
    <jointvel name="s_yd"  joint="slide_y"/>
    <jointpos name="p_hy"  joint="hinge_y"/>
    <jointvel name="p_hyd" joint="hinge_y"/>
    <jointpos name="p_hx"  joint="hinge_x"/>
    <jointvel name="p_hxd" joint="hinge_x"/>
  </sensor>
</mujoco>
"""

# ================================================================
# LQR 控制器（两轴解耦，各自独立 LQR）
# ================================================================
class LQRController:
    """
    XZ 平面（由 hinge_y 控制，force_x 执行）和
    YZ 平面（由 hinge_x 控制，force_y 执行）完全对称解耦。

    单轴状态：[cart_pos, cart_vel, pole_angle, pole_avel]
    控制律：  u = -K @ state
    """

    def __init__(self, use_scipy: bool = True):
        mc  = 2.0     # 小车质量
        mp  = 0.20    # 杆+球质量
        l   = 0.7     # 质心到铰链距离（杆长从0.7增加到1.4）
        g   = 9.81

        den = mc + mp
        A = np.array([
            [0,  1,                    0,  0],
            [0,  0,        -mp * g / den,  0],
            [0,  0,                    0,  1],
            [0,  0,  g * (mc+mp)/(l*den),  0],
        ])
        B = np.array([[0], [1.0/den], [0], [-1.0/(l*den)]])

        Q = np.diag([1.0, 0.5, 12.0, 2.0])
        R = np.array([[0.01]])

        self.K = self._solve_lqr(A, B, Q, R, use_scipy)
        print(f"  LQR 增益 K = {self.K.round(3)}")
        self.max_force = 38.0

    @staticmethod
    def _solve_lqr(A, B, Q, R, use_scipy):
        if use_scipy:
            try:
                from scipy.linalg import solve_continuous_are
                P = solve_continuous_are(A, B, Q, R)
                return (np.linalg.inv(R) @ B.T @ P).flatten()
            except Exception as e:
                print(f"  scipy 失败（{e}），使用预置增益")
        return np.array([-1.0, -2.5, 32.0, 6.5])

    def compute(self, cx, cvx, thy, thyd,
                      cy, cvy, thx, thxd) -> tuple[float, float]:
        """返回 (fx, fy)"""
        def axis(cp, cv, th, thd):
            s    = np.array([cp, cv, th, thd])
            s[2] = (s[2] + np.pi) % (2 * np.pi) - np.pi
            return float(np.clip(-self.K @ s, -self.max_force, self.max_force))

        fx = axis(cx,  cvx, thy, thyd)   # hinge_y 偏角 → force_x
        fy = axis(cy,  cvy, thx, thxd)   # hinge_x 偏角 → force_y
        return fx, fy


# ================================================================
# Camera
# ================================================================
class Camera:

    def __init__(self):
        self.az   = 45.0
        self.el   = -20.0
        self.dist = 9.0
        self.look = np.array([0.0, 0.0, 2.5])
        self.lx = self.ly = None
        self.lb = self.rb = False

    def apply(self, cam):
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth   = self.az
        cam.elevation = self.el
        cam.distance  = self.dist
        cam.lookat[:] = self.look

    def on_button(self, win, btn, act, mods):
        self.lb = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT)  == glfw.PRESS
        self.rb = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if act == glfw.RELEASE:
            self.lx = self.ly = None

    def on_cursor(self, win, x, y):
        if self.lx is None:
            self.lx, self.ly = x, y
            return
        dx, dy = x - self.lx, y - self.ly
        self.lx, self.ly = x, y
        if self.lb:
            self.az -= dx * 0.3
            self.el  = float(np.clip(self.el + dy * 0.3, -89, 89))
        if self.rb:
            s = self.dist * 0.002
            self.look[0] -= dx * s
            self.look[1] += dy * s

    def on_scroll(self, win, xo, yo):
        self.dist = float(np.clip(self.dist * (1 - yo * 0.1), 1, 60))


# ================================================================
# Main
# ================================================================
def main():

    FRAME_DT     = 1.0 / 60.0
    MANUAL_FORCE = 20.0

    model = mujoco.MjModel.from_xml_string(XML)
    data  = mujoco.MjData(model)

    # 关节地址
    jnt = {name: model.jnt_qposadr[
               mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
           for name in ("slide_x", "slide_y", "hinge_y", "hinge_x")}

    act_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "force_x")
    act_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "force_y")

    # 初始扰动（双轴都偏一点）
    data.qpos[jnt["hinge_y"]] =  0.08
    data.qpos[jnt["hinge_x"]] = -0.06

    lqr = LQRController(use_scipy=True)

    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    win = glfw.create_window(1280, 720, "Spherical Cart-Pole", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    cam   = mujoco.MjvCamera()
    opt   = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=500)
    ctx   = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    cam_ctrl = Camera()
    glfw.set_mouse_button_callback(win, cam_ctrl.on_button)
    glfw.set_cursor_pos_callback(win,   cam_ctrl.on_cursor)
    glfw.set_scroll_callback(win,       cam_ctrl.on_scroll)

    key_p_prev = key_r_prev = glfw.RELEASE
    auto_mode  = False

    print("=" * 60)
    print("  W/S       : 手动 Y 轴施力（前/后）")
    print("  A/D       : 手动 X 轴施力（左/右）")
    print("  P         : 切换 LQR 双轴自动平衡")
    print("  R         : 重置（随机双轴扰动）")
    print("  鼠标左键  : 旋转  |  右键: 平移  |  滚轮: 缩放")
    print("  ESC       : 退出")
    print("=" * 60)

    t_print = time.time()
    t_prev  = time.time()

    while not glfw.window_should_close(win):

        t_now   = time.time()
        real_dt = min(t_now - t_prev, 0.05)
        t_prev  = t_now

        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # ------ P 键切换模式 ------
        key_p_now = glfw.get_key(win, glfw.KEY_P)
        if key_p_now == glfw.PRESS and key_p_prev == glfw.RELEASE:
            auto_mode = not auto_mode
            print(f"\n  [模式] {'LQR 双轴自动平衡' if auto_mode else '手动控制'}")
        key_p_prev = key_p_now

        # ------ R 键重置 ------
        key_r_now = glfw.get_key(win, glfw.KEY_R)
        if key_r_now == glfw.PRESS and key_r_prev == glfw.RELEASE:
            mujoco.mj_resetData(model, data)
            data.qpos[jnt["hinge_y"]] = np.random.uniform(-0.2, 0.2)
            data.qpos[jnt["hinge_x"]] = np.random.uniform(-0.2, 0.2)
            data.qvel[:] = 0.0
            print("\n  [重置] 随机双轴扰动")
        key_r_prev = key_r_now

        # ------ 读取传感器 ------
        # sensordata 顺序与 XML <sensor> 顺序一致
        sx, sxd, sy, syd, thy, thyd, thx, thxd = data.sensordata[:8]

        # ------ 控制力计算 ------
        if auto_mode:
            # 归一化后判断是否在线性化有效范围内
            thy_n = (thy + np.pi) % (2 * np.pi) - np.pi
            thx_n = (thx + np.pi) % (2 * np.pi) - np.pi
            if abs(thy_n) < np.radians(65) and abs(thx_n) < np.radians(65):
                fx, fy = lqr.compute(sx, sxd, thy, thyd,
                                     sy, syd, thx, thxd)
            else:
                fx = fy = 0.0
        else:
            fx = fy = 0.0
            if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS: fx -= MANUAL_FORCE
            if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS: fx += MANUAL_FORCE
            if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS: fy -= MANUAL_FORCE
            if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS: fy += MANUAL_FORCE

        data.ctrl[act_x] = fx
        data.ctrl[act_y] = fy

        # ------ 物理步进 ------
        n_steps = max(1, round(real_dt / model.opt.timestep))
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

        # ------ 渲染 ------
        cam_ctrl.apply(cam)
        w, h = glfw.get_framebuffer_size(win)
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), scene, ctx)
        glfw.swap_buffers(win)

        # ------ 状态打印 ------
        if time.time() - t_print > 0.25:
            thy_n  = (thy + np.pi) % (2 * np.pi) - np.pi
            thx_n  = (thx + np.pi) % (2 * np.pi) - np.pi
            tilt   = np.degrees(np.sqrt(thy_n**2 + thx_n**2))
            mode   = "AUTO" if auto_mode else "MANUAL"
            status = "✓ 平衡" if tilt < 12 else ("⚠ 倾斜" if tilt < 80 else "↻ 旋转")
            print(f"\r  [{mode}]  "
                  f"车:({sx:+5.2f},{sy:+5.2f})m  "
                  f"θy:{np.degrees(thy_n):+6.1f}°  "
                  f"θx:{np.degrees(thx_n):+6.1f}°  "
                  f"倾角:{tilt:5.1f}°  "
                  f"F:({fx:+5.1f},{fy:+5.1f})N  {status}   ",
                  end="", flush=True)
            t_print = time.time()

        elapsed = time.time() - t_now
        time.sleep(max(0.0, FRAME_DT - elapsed))

    print()
    ctx.free()
    glfw.terminate()


if __name__ == "__main__":
    main()
