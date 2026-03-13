#!/usr/bin/env python3
"""
Simple Rope Demo
================

简单的绳索物理模拟示例。

演示绳索在重力作用下的摆动效果。
"""

import time
import numpy as np
from rope_sim import RopePhysics, RopeVisualizer


def main():
    # 创建物理引擎
    initial_anchor = np.array([0.0, 0.0, 10.0])
    rope = RopePhysics(anchor=initial_anchor, length=10.0, segment_length=0.1)

    # 创建可视化器
    visualizer = RopeVisualizer(rope, title="Simple Rope Demo")

    dt = 0.01
    t = 0.0

    while visualizer.is_running():
        start_time = time.time()

        # 让锚点做圆周运动
        radius = 2.0
        angular_velocity = 0.5
        anchor = np.array([
            radius * np.sin(angular_velocity * t),
            radius * np.cos(angular_velocity * t),
            10.0 + 0.5 * np.sin(0.3 * t)
        ])

        # 更新物理引擎
        rope.set_anchor(anchor)
        rope.step(dt)

        # 更新可视化
        visualizer.update(anchor)
        visualizer.render()

        # 控制时间
        t += dt
        elapsed = time.time() - start_time
        time.sleep(max(0.0, dt - elapsed))

    visualizer.shutdown()


if __name__ == "__main__":
    main()
