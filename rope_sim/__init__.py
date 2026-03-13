"""
Rope Simulation Library
=======================

A simple rope physics simulation library using Position-Based Dynamics (PBD)
and Verlet integration, with MuJoCo as the visualization backend.

核心模块：
- RopePhysics: 物理引擎，实现绳索的动力学计算
- RopeVisualizer: 可视化器，使用 MuJoCo 渲染物理状态

特点：
- 简单易用的 API
- 物理引擎与可视化完全分离
- 高性能的 PBD 求解器
- 支持参数化配置
"""

from .physics import RopePhysics
from .visualization import RopeVisualizer

__all__ = ["RopePhysics", "RopeVisualizer"]
__version__ = "0.1.0"
