
import mujoco
import mujoco.viewer
import numpy as np
import time

# ─────────────────────────────────────────
# 1  MuJoCo XML 模型
# ─────────────────────────────────────────
XML = """
<mujoco model="crane">

  <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.3"/>
    <material name="cart_mat"  rgba="0.2 0.6 0.9 1"/>
    <material name="rope_mat"  rgba="0.9 0.8 0.1 1"/>
    <material name="ball_mat"  rgba="0.9 0.3 0.2 1"/>
    <material name="rail_mat"  rgba="0.4 0.4 0.4 1"/>
  </asset>

  <worldbody>

    <!-- 地面 -->
    <geom name="floor" type="plane" size="20 20 0.1"
          material="grid" condim="3"/>

    <!-- 导轨 -->
    <geom name="rail_x" type="box" size="5 0.05 0.05" pos="0 0 3"
          material="rail_mat"/>
    <geom name="rail_y" type="box" size="0.05 5 0.05" pos="0 0 3"
          material="rail_mat"/>

    <!-- 平台小车（x/y 自由移动） -->
    <body name="cart" pos="0 0 3">
      <joint name="cart_x" type="slide" axis="1 0 0" range="-4 4"
             damping="5"/>
      <joint name="cart_y" type="slide" axis="0 1 0" range="-4 4"
             damping="5"/>
      <geom name="cart_geom" type="box" size="0.3 0.3 0.1"
            material="cart_mat" mass="200"/>

      <!-- 绳子用 5 节连杆近似 -->
      <body name="rope0" pos="0 0 0">
        <joint name="rope0_x" type="hinge" axis="1 0 0" damping="0.05"/>
        <joint name="rope0_y" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom name="rope0_geom" type="capsule"
              fromto="0 0 0  0 0 -0.2"
              size="0.02" material="rope_mat" mass="0.05"/>

        <body name="rope1" pos="0 0 -0.2">
          <joint name="rope1_x" type="hinge" axis="1 0 0" damping="0.05"/>
          <joint name="rope1_y" type="hinge" axis="0 1 0" damping="0.05"/>
          <geom name="rope1_geom" type="capsule"
                fromto="0 0 0  0 0 -0.2"
                size="0.02" material="rope_mat" mass="0.05"/>

          <body name="rope2" pos="0 0 -0.2">
            <joint name="rope2_x" type="hinge" axis="1 0 0" damping="0.05"/>
            <joint name="rope2_y" type="hinge" axis="0 1 0" damping="0.05"/>
            <geom name="rope2_geom" type="capsule"
                  fromto="0 0 0  0 0 -0.2"
                  size="0.02" material="rope_mat" mass="0.05"/>

            <body name="rope3" pos="0 0 -0.2">
              <joint name="rope3_x" type="hinge" axis="1 0 0" damping="0.05"/>
              <joint name="rope3_y" type="hinge" axis="0 1 0" damping="0.05"/>
              <geom name="rope3_geom" type="capsule"
                    fromto="0 0 0  0 0 -0.2"
                    size="0.02" material="rope_mat" mass="0.05"/>

              <body name="rope4" pos="0 0 -0.2">
                <joint name="rope4_x" type="hinge" axis="1 0 0" damping="0.05"/>
                <joint name="rope4_y" type="hinge" axis="0 1 0" damping="0.05"/>
                <geom name="rope4_geom" type="capsule"
                      fromto="0 0 0  0 0 -0.2"
                      size="0.02" material="rope_mat" mass="0.05"/>

                <!-- 负载球 -->
                <body name="ball" pos="0 0 -0.2">
                  <geom name="ball_geom" type="sphere" size="0.15"
                        material="ball_mat" mass="10"/>
                </body>

              </body>
            </body>
          </body>
        </body>
      </body>

    </body>
  </worldbody>

  <actuator>
    <!-- 平台驱动力 -->
    <motor name="force_x" joint="cart_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
    <motor name="force_y" joint="cart_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
  </actuator>

</mujoco>
"""

# ─────────────────────────────────────────
# 2  键盘状态（非阻塞）
# ─────────────────────────────────────────
try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("[警告] 未安装 keyboard 库，将使用自动演示模式")
    print("       安装方法: pip install keyboard")

# ─────────────────────────────────────────
# 3  控制器
# ─────────────────────────────────────────
class CraneController:
    """
    简单 PD 控制器：
      - 目标位置由 WASD 积分
      - 带轻微 anti-sway 项（负载速度反馈）
    """

    def __init__(self, model, data):
        self.model = model
        self.data  = data

        # 目标位置
        self.target_x = 0.0
        self.target_y = 0.0

        # PD 增益
        self.Kp = 800.0
        self.Kd = 200.0

        # anti-sway 增益
        self.Ks = 60.0

        # 速度指令
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.max_v  = 2.0    # m/s
        self.accel  = 3.0    # m/s²

    # ── 读取 WASD ──────────────────────────
    def read_keys(self, dt):
        if not HAS_KEYBOARD:
            return
        ax = 0.0
        ay = 0.0
        if keyboard.is_pressed('d'): ax += self.accel
        if keyboard.is_pressed('a'): ax -= self.accel
        if keyboard.is_pressed('w'): ay += self.accel
        if keyboard.is_pressed('s'): ay -= self.accel

        self.cmd_vx = np.clip(self.cmd_vx + ax * dt,
                              -self.max_v, self.max_v)
        self.cmd_vy = np.clip(self.cmd_vy + ay * dt,
                              -self.max_v, self.max_v)

        # 松键减速
        if ax == 0: self.cmd_vx *= 0.85
        if ay == 0: self.cmd_vy *= 0.85

        self.target_x += self.cmd_vx * dt
        self.target_y += self.cmd_vy * dt

        # 限位
        self.target_x = np.clip(self.target_x, -3.5, 3.5)
        self.target_y = np.clip(self.target_y, -3.5, 3.5)

    # ── 计算控制力 ─────────────────────────
    def compute(self, dt):
        self.read_keys(dt)

        # 平台当前状态
        cart_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_x")
        cart_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_y")

        qpos_x = self.data.qpos[cart_x_id]
        qpos_y = self.data.qpos[cart_y_id]
        qvel_x = self.data.qvel[cart_x_id]
        qvel_y = self.data.qvel[cart_y_id]

        # 负载位置
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_pos = self.data.xpos[ball_id]
        ball_vel = self.data.cvel[ball_id]   # [6]: [ang vel, lin vel]
        bvx = ball_vel[3]
        bvy = ball_vel[4]

        # PD 误差
        ex = self.target_x - qpos_x
        ey = self.target_y - qpos_y

        # 控制力 = PD + anti-sway
        fx = self.Kp * ex - self.Kd * qvel_x - self.Ks * bvx
        fy = self.Kp * ey - self.Kd * qvel_y - self.Ks * bvy

        return fx, fy

# ─────────────────────────────────────────
# 4  自动演示（无 keyboard 时）
# ─────────────────────────────────────────
class AutoDemo:
    """
    自动走一个方形轨迹，展示 anti-sway 效果
    """
    def __init__(self):
        self.waypoints = [
            ( 2.0,  0.0),
            ( 2.0,  2.0),
            (-2.0,  2.0),
            (-2.0, -2.0),
            ( 0.0,  0.0),
        ]
        self.wp_idx  = 0
        self.timer   = 0.0
        self.hold    = 3.0   # 每个路点停留秒数

    def get_target(self, dt):
        self.timer += dt
        if self.timer > self.hold:
            self.timer  = 0.0
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
        return self.waypoints[self.wp_idx]

# ─────────────────────────────────────────
# 5  主循环
# ─────────────────────────────────────────
def main():
    print("=" * 50)
    print("  MuJoCo 吊车仿真")
    print("=" * 50)
    if HAS_KEYBOARD:
        print("  W/S : Y 轴移动")
        print("  A/D : X 轴移动")
    else:
        print("  自动演示模式（方形轨迹）")
    print("  关闭窗口退出")
    print("=" * 50)

    # 创建模型
    model = mujoco.MjModel.from_xml_string(XML)
    data  = mujoco.MjData(model)

    # 初始化
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    controller = CraneController(model, data)
    demo       = AutoDemo() if not HAS_KEYBOARD else None

    # actuator id
    act_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "force_x")
    act_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "force_y")

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # 摄像机设置
        viewer.cam.distance = 12.0
        viewer.cam.elevation = -25.0
        viewer.cam.azimuth   = 45.0

        sim_time = 0.0

        while viewer.is_running():
            step_start = time.time()

            # ── 自动演示模式 ──────────────────
            if demo is not None:
                tx, ty = demo.get_target(dt)
                controller.target_x = tx
                controller.target_y = ty

            # ── 控制 ──────────────────────────
            fx, fy = controller.compute(dt)
            data.ctrl[act_x] = fx
            data.ctrl[act_y] = fy

            # ── 物理步进 ──────────────────────
            mujoco.mj_step(model, data)
            sim_time += dt

            # ── 状态打印（每秒一次）──────────
            if int(sim_time * 10) % 10 == 0:
                cart_x = data.qpos[0]
                cart_y = data.qpos[1]
                ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                ball_z  = data.xpos[ball_id][2]
                cart_z  = 3.0  # 固定高度
                swing   = cart_z - ball_z
                print(f"t={sim_time:6.2f}s | "
                      f"cart=({cart_x:+.2f},{cart_y:+.2f}) | "
                      f"target=({controller.target_x:+.2f},{controller.target_y:+.2f}) | "
                      f"rope_vertical={swing:.3f}m")

            # ── 同步显示 ──────────────────────
            viewer.sync()

            # ── 实时速率控制 ──────────────────
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
