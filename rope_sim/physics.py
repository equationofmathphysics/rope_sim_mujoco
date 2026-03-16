"""
Rope Physics Engine
===================

基于 Position-Based Dynamics (PBD) 和 Verlet 积分的绳索物理引擎。
纯物理计算模块，无任何可视化代码。
支持绳索伸长和缩短的 active segment 机制。
"""

import numpy as np


class RopePhysics:
    """绳索物理引擎类。

    实现 Position-Based Dynamics (PBD) 绳索模拟，使用 Verlet 积分和距离约束。
    支持绳索伸长和缩短的 active segment 机制。

    Args:
        anchor: 绳索起点锚点坐标 (x, y, z)
        length: 绳索初始长度
        segment_length: 每个绳索段的长度
        max_length: 绳索最大长度（决定最大 segments 数量）
        gravity: 重力加速度向量 (x, y, z)
        iterations: PBD 约束求解的迭代次数
        damping: 阻尼系数（减少弹性）
        end_mass: 末端负载质量（让绳索受重物拉下垂）
        end_inertia: 末端负载的惯性张量（3x3矩阵或对角惯性值）
    """

    def __init__(
        self,
        anchor,
        length: float = 10.0,
        segment_length: float = 0.1,
        max_length: float = 20.0,
        gravity: tuple = (0.0, 0.0, -9.81),
        iterations: int = 8,
        stretch_damping: float = 0.99,  # 伸缩阻尼（轴向阻尼）
        bend_damping: float = 0.98,    # 弯曲/摆动阻尼（垂直于绳索方向）
        node_mass: float = 0.1,         # 每个绳索节点的质量（kg）
        end_mass: float = 0.0,          # 末端重物的质量（kg）
        end_inertia: any = None,        # 末端负载的惯性张量
    ):
        self.segment_length = segment_length
        self.max_segments = int(round(max_length / segment_length))
        self.active_segments = int(round(length / segment_length))
        self.gravity = np.array(gravity, dtype=float)
        self.iterations = iterations
        self.stretch_damping = stretch_damping
        self.bend_damping = bend_damping
        self.node_mass = node_mass
        self.end_mass = end_mass

        # 初始化位置和前一时刻位置（用于 Verlet 积分）
        anchor = np.array(anchor, dtype=float)
        # 初始时让绳子稍微松弛一些，不是完全拉直，这样重物才能拉下垂
        self.positions = np.array(
            [anchor + np.array([0, 0, -i * segment_length * 0.95]) for i in range(self.max_segments)],
            dtype=float,
        )
        self.prev_positions = self.positions.copy()

    @property
    def num_segments(self):
        """向后兼容：获取当前 active segments 数量。"""
        return self.active_segments

    def set_anchor(self, anchor):
        """设置绳索起点锚点位置。

        Args:
            anchor: 新的锚点坐标 (x, y, z)
        """
        self.positions[0] = np.array(anchor, dtype=float)
        self.prev_positions[0] = np.array(anchor, dtype=float)

    def set_length(self, length: float):
        """设置绳索当前长度。

        Args:
            length: 新的绳索长度

        关键细节：当绳索变长时，新的 segments 需要正确初始化，
        避免 Verlet 积分产生巨大速度导致爆炸。
        """
        target_active_segments = int(round(length / self.segment_length))
        target_active_segments = np.clip(target_active_segments, 1, self.max_segments)

        if target_active_segments > self.active_segments:
            # 绳索变长 - 初始化新 segments
            for i in range(self.active_segments, target_active_segments):
                last_pos = self.positions[self.active_segments - 1]
                new_pos = last_pos + np.array([0, 0, -self.segment_length])
                self.positions[i] = new_pos
                self.prev_positions[i] = new_pos.copy()
        elif target_active_segments < self.active_segments:
            # 绳索缩短 - 直接调整 active segments 数量即可
            pass

        self.active_segments = target_active_segments

    def step(self, dt: float):
        """物理引擎步进。

        Args:
            dt: 时间步长
        """
        # Verlet 积分更新位置（只更新 active segments）
        for i in range(1, self.active_segments):
            velocity = self.positions[i] - self.prev_positions[i]

            # 应用各向异性阻尼：区分轴向（伸缩）和横向（摆动）
            if i < self.active_segments - 1:
                # 计算绳索局部方向（轴向）
                if i > 0:
                    rope_dir = self.positions[i + 1] - self.positions[i - 1]
                    rope_len = np.linalg.norm(rope_dir)
                    if rope_len > 1e-8:
                        rope_dir = rope_dir / rope_len
                        # 分解速度到轴向和横向
                        vel_stretch = np.dot(velocity, rope_dir) * rope_dir
                        vel_bend = velocity - vel_stretch
                        # 分别应用阻尼
                        velocity = vel_stretch * self.stretch_damping + vel_bend * self.bend_damping
            else:
                # 末端点直接用弯曲阻尼（主要是摆动）
                velocity *= self.bend_damping

            self.prev_positions[i] = self.positions[i].copy()

            # 所有节点都受到相同的重力加速度
            # 质量的影响体现在约束求解时的惯性权重上
            self.positions[i] += velocity + self.gravity * dt * dt

        # PBD 距离约束求解（只求解 active segments）
        for _ in range(self.iterations):
            for i in range(self.active_segments - 1):
                delta = self.positions[i + 1] - self.positions[i]
                distance = np.linalg.norm(delta)

                if distance < 1e-8:
                    continue

                correction = delta * ((distance - self.segment_length) / distance)

                if i == 0:
                    # 锚点固定，只移动另一端
                    self.positions[i + 1] -= correction
                else:
                    # 根据节点质量计算惯性权重
                    # 质量越大，惯性越大，移动越少
                    if i == self.active_segments - 2:
                        # 最后一个约束：连接到末端重物
                        mass_node = self.node_mass
                        mass_end = self.node_mass + self.end_mass

                        weight_node = 1.0 / mass_node
                        weight_end = 1.0 / mass_end

                        total_weight = weight_node + weight_end

                        ratio_node = weight_node / total_weight
                        ratio_end = weight_end / total_weight

                        self.positions[i] += correction * ratio_node
                        self.positions[i + 1] -= correction * ratio_end
                    else:
                        # 中间节点质量相同
                        self.positions[i] += correction * 0.5
                        self.positions[i + 1] -= correction * 0.5
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    @property
    def end_point(self):
        """获取绳索终点位置。

        Returns:
            绳索终点坐标 (x, y, z)
        """
        return self.positions[self.active_segments - 1].copy()

    def get_segment_positions(self):
        """获取所有 active 绳索段的位置。

        Returns:
            每个绳索段的起点和终点位置数组
        """
        return np.array([(self.positions[i], self.positions[i + 1]) for i in range(self.active_segments - 1)])
