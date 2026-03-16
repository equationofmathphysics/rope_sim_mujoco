"""
Rope Simulation with MuJoCo Joints
===================================

基于 MuJoCo 关节系统的绳索仿真模块。

这是工业级仿真的正道：把绳索当作"最大长度可变、只能受拉不能受压的弹簧（或刚性连杆）"，
把运算资源留给刚体动力学。
"""

from .rope_sim_Joint import RopeJointSim, RopeChainSim

__all__ = ['RopeJointSim', 'RopeChainSim']
