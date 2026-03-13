"""
Rope Physics Engine
===================

基于 Position-Based Dynamics (PBD) 和 Verlet 积分的绳索物理引擎。
纯物理计算模块，无任何可视化代码。
"""

import numpy as np


class RopePhysics:
    """绳索物理引擎类。

    实现 Position-Based Dynamics (PBD) 绳索模拟，使用 Verlet 积分和距离约束。

    Args:
        anchor: 绳索起点锚点坐标 (x, y, z)
        length: 绳索总长度
        segment_length: 每个绳索段的长度
        gravity: 重力加速度向量 (x, y, z)
        iterations: PBD 约束求解的迭代次数
    """

    def __init__(
        self,
        anchor,
        length: float = 10.0,
        segment_length: float = 0.1,
        gravity: tuple = (0.0, 0.0, -9.81),
        iterations: int = 8,
    ):
        self.segment_length = segment_length
        self.num_segments = int(round(length / segment_length))
        self.gravity = np.array(gravity, dtype=float)
        self.iterations = iterations

        # 初始化位置和前一时刻位置（用于 Verlet 积分）
        self.positions = np.array(
            [anchor + np.array([0, 0, -i * segment_length]) for i in range(self.num_segments)],
            dtype=float,
        )
        self.prev_positions = self.positions.copy()

    def set_anchor(self, anchor):
        """设置绳索起点锚点位置。

        Args:
            anchor: 新的锚点坐标 (x, y, z)
        """
        self.positions[0] = np.array(anchor, dtype=float)
        self.prev_positions[0] = np.array(anchor, dtype=float)

    def step(self, dt: float):
        """物理引擎步进。

        Args:
            dt: 时间步长
        """
        # Verlet 积分更新位置
        for i in range(1, self.num_segments):
            velocity = self.positions[i] - self.prev_positions[i]
            self.prev_positions[i] = self.positions[i].copy()
            self.positions[i] += velocity + self.gravity * dt * dt

        # PBD 距离约束求解
        for _ in range(self.iterations):
            for i in range(self.num_segments - 1):
                delta = self.positions[i + 1] - self.positions[i]
                distance = np.linalg.norm(delta)

                if distance < 1e-8:
                    continue

                correction = delta * ((distance - self.segment_length) / distance)

                if i == 0:
                    self.positions[i + 1] -= correction
                else:
                    self.positions[i] += correction * 0.5
                    self.positions[i + 1] -= correction * 0.5

    @property
    def end_point(self):
        """获取绳索终点位置。

        Returns:
            绳索终点坐标 (x, y, z)
        """
        return self.positions[-1].copy()

    def get_segment_positions(self):
        """获取所有绳索段的位置。

        Returns:
            每个绳索段的起点和终点位置数组
        """
        return np.array([(self.positions[i], self.positions[i + 1]) for i in range(self.num_segments - 1)])
