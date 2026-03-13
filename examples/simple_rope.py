#!/usr/bin/env python3
"""
Simple Rope Demo
================

简单的绳索物理模拟示例。

演示绳索在重力作用下的摆动效果，以及绳索伸长和缩短。

控制方式：
    A / D      → 锚点水平移动
    W / S      → 绳索伸长/缩短
    左键拖动   → 旋转视角
    右键拖动   → 平移视角
    滚轮       → 缩放
    ESC        → 退出
"""

import time
import numpy as np
import glfw
from rope_sim import RopePhysics, RopeVisualizer


def main():
    print("=" * 50)
    print("  Simple Rope Demo - with rope extend/retract")
    print("=" * 50)
    print("  A / D      → 锚点水平移动")
    print("  W / S      → 绳索伸长/缩短")
    print("  左键拖动   → 旋转视角")
    print("  右键拖动   → 平移视角")
    print("  滚轮       → 缩放")
    print("  ESC        → 退出")
    print("=" * 50)

    # 物理引擎参数
    INITIAL_LENGTH = 10.0
    MAX_LENGTH = 20.0
    MIN_LENGTH = 3.0
    MOVE_SPEED = 0.05
    LENGTH_SPEED = 0.1

    # 创建物理引擎
    initial_anchor = np.array([0.0, 0.0, 12.0])
    rope = RopePhysics(
        anchor=initial_anchor,
        length=INITIAL_LENGTH,
        segment_length=0.1,
        max_length=MAX_LENGTH,
        iterations=8
    )

    # 创建可视化器
    visualizer = RopeVisualizer(rope, title="Simple Rope Demo")

    dt = 0.01
    t = 0.0
    current_length = INITIAL_LENGTH
    anchor_x = 0.0

    while visualizer.is_running():
        start_time = time.time()

        # 键盘控制
        if visualizer.get_key_state(glfw.KEY_A):
            anchor_x -= MOVE_SPEED
        if visualizer.get_key_state(glfw.KEY_D):
            anchor_x += MOVE_SPEED
        if visualizer.get_key_state(glfw.KEY_W):
            current_length += LENGTH_SPEED
        if visualizer.get_key_state(glfw.KEY_S):
            current_length -= LENGTH_SPEED
        if visualizer.get_key_state(glfw.KEY_ESCAPE):
            break

        # 限制长度范围
        current_length = float(np.clip(current_length, MIN_LENGTH, MAX_LENGTH))

        # 更新锚点 - 做简单的摆动
        anchor = np.array([anchor_x, 0.0, 12.0])

        # 更新物理引擎
        rope.set_anchor(anchor)
        rope.set_length(current_length)
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
