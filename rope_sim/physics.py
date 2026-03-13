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
    """

    def __init__(
        self,
        anchor,
        length: float = 10.0,
        segment_length: float = 0.1,
        max_length: float = 20.0,
        gravity: tuple = (0.0, 0.0, -9.81),
        iterations: int = 8,
    ):
        self.segment_length = segment_length
        self.max_segments = int(round(max_length / segment_length))
        self.active_segments = int(round(length / segment_length))
        self.gravity = np.array(gravity, dtype=float)
        self.iterations = iterations

        # 初始化位置和前一时刻位置（用于 Verlet 积分）
        anchor = np.array(anchor, dtype=float)
        self.positions = np.array(
            [anchor + np.array([0, 0, -i * segment_length]) for i in range(self.max_segments)],
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
            self.prev_positions[i] = self.positions[i].copy()
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
        return self.positions[self.active_segments - 1].copy()

    def get_segment_positions(self):
        """获取所有 active 绳索段的位置。

        Returns:
            每个绳索段的起点和终点位置数组
        """
        return np.array([(self.positions[i], self.positions[i + 1]) for i in range(self.active_segments - 1)])
