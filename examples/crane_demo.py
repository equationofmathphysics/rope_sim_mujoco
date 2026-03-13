#!/usr/bin/env python3
"""
Crane Rope Demo
===============

演示起重机绳索的物理模拟。

控制方式：
    A / D      → 轨道 Y 轴
    W / S      → 小车 X 轴
    Q / E      → 升降
    左键拖动   → 旋转视角
    右键拖动   → 平移视角
    滚轮       → 缩放
    ESC        → 退出
"""

import time
import glfw
import numpy as np
from rope_sim import RopePhysics, RopeVisualizer


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

    # 物理引擎参数
    HOIST_HEIGHT = 12.0
    MOVE_SPEED = 0.05

    # 创建物理引擎
    initial_anchor = np.array([0.0, 0.0, HOIST_HEIGHT])
    rope = RopePhysics(anchor=initial_anchor, length=10.0, segment_length=0.1, iterations=8)

    # 创建可视化器
    visualizer = RopeVisualizer(rope, title="Crane Rope Demo")

    # 起重机状态
    track_y = 0.0
    cabin_x = 0.0
    hoist_z = 0.0
    dt = float(visualizer.model.opt.timestep)

    while visualizer.is_running():
        start_time = time.time()

        # 键盘控制
        if visualizer.get_key_state(glfw.KEY_A):
            track_y += MOVE_SPEED
        if visualizer.get_key_state(glfw.KEY_D):
            track_y -= MOVE_SPEED
        if visualizer.get_key_state(glfw.KEY_W):
            cabin_x += MOVE_SPEED
        if visualizer.get_key_state(glfw.KEY_S):
            cabin_x -= MOVE_SPEED
        if visualizer.get_key_state(glfw.KEY_Q):
            hoist_z += MOVE_SPEED
        if visualizer.get_key_state(glfw.KEY_E):
            hoist_z -= MOVE_SPEED
        if visualizer.get_key_state(glfw.KEY_ESCAPE):
            break

        # 限制升降范围
        hoist_z = float(np.clip(hoist_z, 0.0, HOIST_HEIGHT - 1.0))
        anchor = np.array([cabin_x, track_y, HOIST_HEIGHT - hoist_z])

        # 更新物理引擎
        rope.set_anchor(anchor)
        rope.step(dt)

        # 更新可视化
        visualizer.update(anchor)
        visualizer.render()

        # 控制帧率
        elapsed = time.time() - start_time
        time.sleep(max(0.0, dt - elapsed))

    visualizer.shutdown()


if __name__ == "__main__":
    main()
