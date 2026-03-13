"""
Utility Functions
=================

通用工具函数模块，包含四元数计算、XML 生成等辅助功能。
"""

import numpy as np
import mujoco


def quat_from_z_to_vec(v):
    """计算使 Z 轴对齐到指定向量的四元数。

    Args:
        v: 目标向量

    Returns:
        四元数 (w, x, y, z)
    """
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])

    v = v / norm
    z_axis = np.array([0.0, 0.0, 1.0])
    dot_product = float(np.clip(np.dot(z_axis, v), -1.0, 1.0))

    if dot_product > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot_product < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])

    axis = np.cross(z_axis, v)
    axis /= np.linalg.norm(axis)
    half_angle = np.arccos(dot_product) / 2.0
    sin_half = np.sin(half_angle)

    return np.array([np.cos(half_angle), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half])


def build_rope_xml(num_segments, radius=0.02, segment_length=0.1, color="0.15 0.15 0.15 1"):
    """动态生成 MuJoCo 绳索模型的 XML 字符串。

    Args:
        num_segments: 绳索分段数量
        radius: 绳索半径
        segment_length: 每个分段的长度
        color: 绳索颜色 (rgba)

    Returns:
        MuJoCo XML 字符串
    """
    rope_bodies = ""
    for i in range(num_segments - 1):
        rope_bodies += f"""
        <body name="rope_{i}" mocap="true">
            <geom type="capsule"
                  size="{radius} {segment_length * 0.5}"
                  rgba="{color}"
                  contype="0" conaffinity="0"/>
        </body>"""

    return f"""
<mujoco model="rope_sim">

    <option timestep="0.01" gravity="0 0 -9.81"/>

    <visual>
        <map znear="0.05" zfar="500"/>
        <global offwidth="1280" offheight="720"/>
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


def get_mocap_id(model, name):
    """获取 MuJoCo 模型中 mocap 体的 ID。

    Args:
        model: MuJoCo 模型对象
        name: body 名称

    Returns:
        mocap 体的 ID
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return model.body_mocapid[body_id]
