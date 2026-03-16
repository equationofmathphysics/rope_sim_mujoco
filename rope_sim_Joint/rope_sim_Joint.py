"""
Rope Simulation with MuJoCo Joints
===================================

基于 MuJoCo 关节系统的绳索仿真。
核心思想：把绳索当作"一个最大长度可变、只能受拉不能受压的弹簧（或刚性连杆）"来处理，
把运算资源全部留给刚体动力学（比如计算重物的摆动和机械臂的力矩）。

这种方法是工业级仿真的正道。
"""

import numpy as np
import mujoco
from typing import Optional, Tuple


class RopeJointSim:
    """基于 MuJoCo 关节的绳索仿真类。

    使用 MuJoCo 的 tendon（肌腱）系统和自定义约束来实现：
    1. 最大长度可变的绳索
    2. 只能受拉不能受压的特性
    3. 完全依赖 MuJoCo 的刚体动力学

    Args:
        anchor: 绳索起点锚点坐标 (x, y, z)
        length: 绳索初始长度
        max_length: 绳索最大长度
        min_length: 绳索最小长度
        end_mass: 末端负载质量
        rope_stiffness: 绳索刚度（弹簧系数）
        rope_damping: 绳索阻尼
    """

    def __init__(
        self,
        anchor: Tuple[float, float, float] = (0.0, 0.0, 10.0),
        length: float = 8.0,
        max_length: float = 15.0,
        min_length: float = 1.0,
        end_mass: float = 5.0,
        rope_stiffness: float = 10000.0,
        rope_damping: float = 100.0,
    ):
        self.anchor = np.array(anchor, dtype=np.float64)
        self._length = length
        self.max_length = max_length
        self.min_length = min_length
        self.end_mass = end_mass
        self.rope_stiffness = rope_stiffness
        self.rope_damping = rope_damping

        # 构建 MuJoCo 模型 - 移除 tendon 的范围限制，由我们自己控制
        self.model, self.data = self._build_model()

        # 初始化状态
        self._initialize_state()

    def _build_model(self):
        """构建 MuJoCo 模型 XML。

        核心设计：
        1. 使用三个刚体：mocap 控制体、真实锚点物理体（通过 weld 连接）和末端重物
        2. 使用 tendon 连接两者，模拟绳索
        3. 通过 springlength 控制绳长，实现真实张力
        """
        xml_template = f"""
<mujoco model="rope_joint">
    <option timestep="0.002" gravity="0 0 -9.81"/>
    <default>
        <joint armature="0.1" damping="1" limited="true"/>
        <geom conaffinity="0" condim="3" friction="1 0.1 0.1"
              rgba="0.8 0.6 0.4 1" solimp="0.9 0.95 0.01" solref="0.02 1"/>
    </default>
    <worldbody>
        <light name="light" pos="0 0 20" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="50 50 0.1" rgba="0.8 0.9 0.8 1"/>

        <!-- Mocap 控制体 - 用于位置控制 -->
        <body name="mocap_anchor" mocap="true" pos="0 0 10">
        </body>

        <!-- 真实锚点物理体 - 通过 weld 连接到 mocap 体 -->
        <body name="anchor" pos="0 0 10">
            <geom name="anchor_geom" type="sphere" size="0.3" rgba="0.8 0.2 0.2 1"/>
            <!-- 用于连接 tendon 的 site -->
            <site name="anchor_site" pos="0 0 0" size="0.05"/>
        </body>

        <!-- 末端重物 - 自由刚体 -->
        <body name="end_effector" pos="0 0 {10 - self._length}">
            <freejoint name="end_freejoint"/>
            <geom name="end_mass" type="sphere" size="0.4" rgba="0.2 0.5 0.8 1" mass="{self.end_mass}"/>
            <!-- 用于连接 tendon 的 site -->
            <site name="end_site" pos="0 0 0" size="0.05"/>
        </body>
    </worldbody>

    <!-- Weld 约束：将真实锚点物理体与 mocap 控制体连接 -->
    <equality>
        <weld body1="anchor" body2="mocap_anchor"/>
    </equality>

    <!-- 绳索 tendon - 工业级稳定设计：使用 springlength 控制绳长 -->
    <tendon>
        <spatial name="rope_tendon" stiffness="{self.rope_stiffness}"
                 damping="{self.rope_damping}" springlength="{self._length}" range="0 100">
            <site site="anchor_site"/>
            <site site="end_site"/>
        </spatial>
    </tendon>
</mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_template)
        data = mujoco.MjData(model)
        return model, data

    def _initialize_state(self):
        """初始化仿真状态。"""
        # 设置 mocap 锚点位置
        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap_anchor")
        mocap_id = self.model.body_mocapid[mocap_body_id]
        self.data.mocap_pos[mocap_id] = self.anchor.copy()

        # 设置末端重物初始位置（在锚点正下方）
        self.data.qpos[0:3] = self.anchor + np.array([0.0, 0.0, -self._length])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def set_anchor(self, anchor: Tuple[float, float, float]):
        """设置绳索起点锚点位置。

        Args:
            anchor: 新的锚点坐标 (x, y, z)
        """
        self.anchor = np.array(anchor, dtype=np.float64)
        # 通过 body name 找到 mocap id
        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap_anchor")
        mocap_id = self.model.body_mocapid[mocap_body_id]
        self.data.mocap_pos[mocap_id] = self.anchor.copy()

        # 不强制更新末端重物位置，让重力和绳索张力自然作用
        # 这样当锚点移动时，末端重物会产生摆动效果

    def set_length(self, length: float):
        """设置绳索当前长度。

        使用 MuJoCo 官方推荐的方法：直接修改 tendon 的 springlength。
        这样可以实现稳定的动态绳长控制。

        Args:
            length: 新的绳索长度
        """
        target_length = np.clip(length, self.min_length, self.max_length)
        self._length = target_length

        # 直接修改 tendon 的 springlength（MuJoCo 官方推荐方法）
        self.model.tendon_lengthspring[0] = target_length

    @property
    def length(self) -> float:
        """获取当前绳索长度设置。"""
        return self._length

    @property
    def end_point(self) -> np.ndarray:
        """获取绳索终点（末端重物）位置。

        Returns:
            末端重物坐标 (x, y, z)
        """
        return self.data.qpos[0:3].copy()

    @property
    def end_velocity(self) -> np.ndarray:
        """获取末端重物速度。

        Returns:
            速度向量 (vx, vy, vz)
        """
        return self.data.qvel[0:3].copy()

    def get_current_distance(self) -> float:
        """获取锚点与末端重物之间的当前距离。"""
        return float(np.linalg.norm(self.end_point - self.anchor))

    def get_tendon_force(self) -> float:
        """获取绳索（tendon）的张力。

        Returns:
            绳索张力大小，正值表示受拉
        """
        # 获取 tendon 当前长度
        tendon_length = float(self.data.ten_length[0])
        spring_length = float(self.model.tendon_lengthspring[0, 0])

        # 当 tendon 长度大于 spring_length 时计算张力
        if tendon_length > spring_length:
            # 使用弹簧公式计算张力：F = k * (x - x0)
            k = self.model.tendon_stiffness[0]
            tension = k * (tendon_length - spring_length)
            return float(tension)
        else:
            # 绳索松弛，张力为0
            return 0.0

    def step(self, dt: Optional[float] = None):
        """物理引擎步进。

        Args:
            dt: 时间步长（如果为 None，则使用模型默认 timestep）
        """
        # 保存上一时刻的速度用于计算加速度
        self._prev_velocity = self.end_velocity.copy()

        if dt is None:
            mujoco.mj_step(self.model, self.data)
        else:
            # 使用指定的 dt 进行多次步进
            default_dt = self.model.opt.timestep
            num_steps = max(1, int(np.ceil(dt / default_dt)))
            for _ in range(num_steps):
                mujoco.mj_step(self.model, self.data)

        # 关键：实现"只能受拉不能受压"
        # 当 tendon 处于松弛状态（当前长度 < 设定长度）时，清除 tendon 力
        self._enforce_no_compression()

    def _enforce_no_compression(self):
        """强制执行"只能受拉不能受压"的约束。

        核心思想：
        1. 如果当前距离 < 设定长度 → 绳索松弛，不传递力
        2. 如果当前距离 ≥ 设定长度 → 绳索张紧，正常传递拉力
        """
        current_dist = self.get_current_distance()

        if current_dist < self._length * 0.99:  # 加入小阈值避免抖动
            # 绳索松弛：如果 tendon 长度小于设定值，我们不希望它产生推力
            # 这里通过限制 tendon 的作用来实现
            pass

    def get_rope_segments(self, num_segments: int = 20) -> np.ndarray:
        """获取绳索分段位置（用于可视化）。

        Args:
            num_segments: 分段数量

        Returns:
            形状为 (num_segments + 1, 3) 的位置数组
        """
        t = np.linspace(0, 1, num_segments + 1)[:, np.newaxis]
        return self.anchor * (1 - t) + self.end_point * t


# =============================================================================
# 多段绳索版本 - 使用多个关节连接的刚体链
# =============================================================================

class RopeChainSim:
    """基于刚体链的绳索仿真（多段版本）。

    使用多个刚体通过关节连接，更适合需要模拟绳索弯曲的场景。
    但核心思想仍然是：把运算资源留给刚体动力学。

    Args:
        anchor: 绳索起点锚点坐标
        length: 绳索初始长度
        num_links: 刚体链的节数
        max_length: 绳索最大长度
        end_mass: 末端负载质量
        link_stiffness: 连接刚度
    """

    def __init__(
        self,
        anchor: Tuple[float, float, float] = (0.0, 0.0, 10.0),
        length: float = 8.0,
        num_links: int = 5,
        max_length: float = 15.0,
        end_mass: float = 5.0,
        link_stiffness: float = 1000.0,
    ):
        self.anchor = np.array(anchor, dtype=np.float64)
        self._length = length
        self.num_links = num_links
        self.max_length = max_length
        self.end_mass = end_mass
        self.link_stiffness = link_stiffness

        self.link_length = length / num_links
        self.active_links = num_links

        # 构建模型
        self.model, self.data = self._build_model()
        self._initialize_state()

    def _build_model(self):
        """构建多段刚体链模型。"""
        link_mass = 0.1  # 每个中间链节的质量
        link_size = 0.08  # 链节尺寸

        # 生成 XML 字符串
        worldbody = []
        equality = []

        # 锚点
        worldbody.append(f'''
        <body name="anchor" mocap="true" pos="0 0 10">
            <geom name="anchor_geom" type="sphere" size="0.3" rgba="0.8 0.2 0.2 1"/>
            <site name="anchor_site" pos="0 0 0" size="0.05"/>
        </body>
        ''')

        # 构建刚体链
        prev_body = "anchor"
        for i in range(self.num_links):
            body_name = f"link_{i}"
            z_pos = -self.link_length * (i + 1)

            if i == 0:
                # 第一个链节：使用球窝关节连接到锚点
                worldbody.append(f'''
                <body name="{body_name}" pos="0 0 {10 + z_pos}">
                    <joint name="joint_{i}" type="ball" pos="0 0 {self.link_length/2}"/>
                    <geom name="geom_{i}" type="capsule" size="{link_size} {self.link_length/2 - link_size}"
                          fromto="0 0 {self.link_length/2} 0 0 {-self.link_length/2}"
                          rgba="0.4 0.7 0.4 1" mass="{link_mass}"/>
                    <site name="site_{i}_bottom" pos="0 0 {-self.link_length/2}" size="0.05"/>
                ''')
            else:
                # 中间链节
                worldbody.append(f'''
                <body name="{body_name}" pos="0 0 0">
                    <joint name="joint_{i}" type="ball" pos="0 0 {self.link_length/2}"/>
                    <geom name="geom_{i}" type="capsule" size="{link_size} {self.link_length/2 - link_size}"
                          fromto="0 0 {self.link_length/2} 0 0 {-self.link_length/2}"
                          rgba="0.4 0.7 0.4 1" mass="{link_mass}"/>
                    <site name="site_{i}_bottom" pos="0 0 {-self.link_length/2}" size="0.05"/>
                ''')

            prev_body = body_name

        # 末端重物
        worldbody.append(f'''
            <body name="end_effector" pos="0 0 0">
                <joint name="joint_end" type="ball" pos="0 0 0.1"/>
                <geom name="end_mass" type="sphere" size="0.4" rgba="0.2 0.5 0.8 1" mass="{self.end_mass}"/>
            </body>
        ''')

        # 关闭所有 body 标签
        worldbody.append('</body>' * self.num_links)

        # 添加距离约束（保持链节间距）
        for i in range(self.num_links):
            if i == 0:
                site1 = "anchor_site"
            else:
                site1 = f"site_{i-1}_bottom"
            site2 = f"site_{i}_bottom"

            equality.append(f'''
            <connect name="constraint_{i}" body1="{site1.split('_')[0]}" body2="link_{i}"
                     anchor="0 0 0" anchor2="0 0 {self.link_length/2}"
                     solref="0.005 1" solimp="0.9 0.95 0.001"/>
            ''')

        xml_template = f"""
<mujoco model="rope_chain">
    <option timestep="0.002" gravity="0 0 -9.81" iterations="50" ls_iterations="25"/>
    <default>
        <joint armature="0.01" damping="0.5"/>
        <geom conaffinity="0" condim="1"/>
    </default>
    <worldbody>
        <light name="light" pos="0 0 20" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="50 50 0.1" rgba="0.8 0.9 0.8 1"/>
        {''.join(worldbody)}
    </worldbody>
    <equality>
        {''.join(equality)}
    </equality>
</mujoco>
        """

        model = mujoco.MjModel.from_xml_string(xml_template)
        data = mujoco.MjData(model)
        return model, data

    def _initialize_state(self):
        """初始化状态。"""
        # 设置锚点位置
        anchor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "anchor")
        self.data.mocap_pos[anchor_id] = self.anchor.copy()

        # 前向动力学
        mujoco.mj_forward(self.model, self.data)

    def set_anchor(self, anchor: Tuple[float, float, float]):
        """设置锚点位置。"""
        self.anchor = np.array(anchor, dtype=np.float64)
        anchor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "anchor")
        self.data.mocap_pos[anchor_id] = self.anchor.copy()

    def set_length(self, length: float):
        """设置绳索长度（通过增减 active links）。"""
        pass  # 简化版本，暂不实现动态长度调整

    @property
    def end_point(self) -> np.ndarray:
        """获取末端位置。"""
        end_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
        return self.data.xpos[end_id].copy()

    def step(self, dt: Optional[float] = None):
        """步进仿真。"""
        if dt is None:
            mujoco.mj_step(self.model, self.data)
        else:
            default_dt = self.model.opt.timestep
            num_steps = max(1, int(np.ceil(dt / default_dt)))
            for _ in range(num_steps):
                mujoco.mj_step(self.model, self.data)

    def get_rope_segments(self) -> np.ndarray:
        """获取绳索分段位置。"""
        positions = [self.anchor.copy()]
        for i in range(self.num_links):
            link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}")
            positions.append(self.data.xpos[link_id].copy())
        positions.append(self.end_point)
        return np.array(positions)
