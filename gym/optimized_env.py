import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from optimized_engine import Point, Config
from optimized_renderer import Renderer
from optimized_walker import Creature, Muscle, Skeleton
import time

class PhysicsEnv:
    """基于物理模拟的环境，类似于OpenAI Gym的接口"""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }
    
    def __init__(self, creature: Creature, in3d: bool = False, g: float = 100, 
                 dampk: float = 0, ground_high: float = 0, ground_k: float = 1000, 
                 ground_damp: float = 100, friction: float = 100, rand_sigma: float = 0.1):
        """
        初始化物理环境
        
        :param creature: 环境中的生物
        :param in3d: 是否使用3D物理
        :param g: 重力加速度
        :param dampk: 阻尼系数
        :param ground_high: 地面高度
        :param ground_k: 地面弹性系数
        :param ground_damp: 地面阻尼系数
        :param friction: 地面摩擦系数
        :param rand_sigma: 随机初始速度的标准差
        """
        self.creature = creature
        self.in3d = in3d
        self.g = g
        self.dampk = dampk
        self.ground = ground_high
        self.ground_k = ground_k
        self.ground_damp = ground_damp
        self.friction = friction
        self.sigma = rand_sigma
        
        # 环境状态
        self.time_step = 0.01
        self.steps = 0
        self.max_steps = 1000
        
        # 渲染相关
        self.renderer = None
        self.render_mode = None
        
        # 重置环境
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        # 清除所有点的加速度
        for p in self.creature.phys:
            p.zero()
            # 添加随机初始速度
            p.v[0] += np.random.normal(0, self.sigma)
            p.v[1] += np.random.normal(0, self.sigma)
            if self.in3d:
                p.v[2] += np.random.normal(0, self.sigma)
        
        # 重置计数器
        self.steps = 0
        
        # 返回初始观察
        return self._get_observation()
    
    def step(self, action: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步环境模拟
        
        :param action: 动作，控制生物的肌肉
        :return: 观察、奖励、是否终止、额外信息
        """
        # 应用动作
        self.creature.act(action)
        
        # 运行物理模拟
        self._run_physics()
        
        # 增加步数
        self.steps += 1
        
        # 计算观察、奖励和终止条件
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        渲染环境
        
        :param mode: 渲染模式，'human'或'rgb_array'
        :return: 如果是'rgb_array'模式，返回RGB图像数组
        """
        self.render_mode = mode
        
        if self.renderer is None:
            self.renderer = Renderer()
        
        # 处理事件
        self.renderer.handle_events()
        
        # 渲染画面
        self.renderer.render()
        self.renderer.tick(60)
        
        # 如果不是运行状态，关闭渲染器
        if not self.renderer.is_running():
            self.close()
            return None
        
        # 如果是rgb_array模式，返回图像
        if mode == 'rgb_array':
            return pygame.surfarray.array3d(self.renderer.screen)
        
        return None
    
    def close(self) -> None:
        """关闭环境"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        设置随机数种子
        
        :param seed: 随机数种子
        :return: 种子列表
        """
        np.random.seed(seed)
        return [seed] if seed is not None else []
    
    def _run_physics(self) -> None:
        """运行物理模拟"""
        # 运行生物的物理模拟
        self.creature.run()
        
        # 应用全局力
        for p in self.creature.phys:
            # 应用重力
            p.forced([0, -self.g, 0] if self.in3d else [0, -self.g])
            
            # 应用阻尼
            self._damp(p, self.dampk)
            
            # 检测地面碰撞
            if p.pos[1] - self.ground < 0:
                p.color = "red"  # 碰撞时变红
                p.r = 3  # 碰撞时变大
                
                # 计算穿透深度
                deep = p.pos[1] - self.ground
                
                # 应用地面反弹力
                p.forced([0, -self.ground_k * deep, 0] if self.in3d else [0, -self.ground_k * deep])
                
                # 应用地面阻尼
                p.forced([0, -self.ground_damp * p.v[1], 0] if self.in3d else [0, -self.ground_damp * p.v[1]])
                
                # 应用地面摩擦力
                friction_force = np.abs(deep) * self.friction
                if self.in3d:
                    p.forced([-p.v[0] * friction_force, 0, -p.v[2] * friction_force])
                else:
                    p.forced([-p.v[0] * friction_force, 0])
            else:
                p.color = "black"  # 正常状态为黑色
                p.r = 1  # 正常状态大小
        
        # 更新物理状态
        Point.run1(self.time_step)
    
    def _damp(self, p: Point, k: float) -> None:
        """应用阻尼力"""
        p.forced(-k * p.v)
    
    def _get_observation(self) -> np.ndarray:
        """获取环境观察"""
        # 使用生物的状态作为观察
        return np.array(self.creature.getstat(self.in3d))
    
    def _get_reward(self) -> float:
        """计算奖励"""
        # 简单的奖励函数：基于生物的高度和稳定性
        # 计算质心高度
        centroid_y = np.mean([p.pos[1] for p in self.creature.phys])
        
        # 计算速度惩罚（过快的速度会获得负奖励）
        avg_velocity = np.mean([np.linalg.norm(p.v) for p in self.creature.phys])
        velocity_penalty = -avg_velocity * 0.1
        
        # 计算碰撞惩罚
        collision_penalty = -sum(1 for p in self.creature.phys if p.pos[1] - self.ground < 0) * 0.5
        
        # 总奖励
        reward = centroid_y + velocity_penalty + collision_penalty
        
        return reward
    
    def _is_done(self) -> bool:
        """判断是否终止"""
        # 检查是否达到最大步数
        if self.steps >= self.max_steps:
            return True
        
        # 检查生物是否过于倾斜或崩溃
        # 计算质心高度
        centroid_y = np.mean([p.pos[1] for p in self.creature.phys])
        
        # 如果质心过低，认为生物已经崩溃
        if centroid_y < self.ground - 50:
            return True
        
        # 检查是否所有点都静止不动（可能卡住了）
        all_stopped = all(np.linalg.norm(p.v) < 0.1 for p in self.creature.phys)
        if all_stopped and self.steps > 100:
            return True
        
        # 检查渲染器是否关闭
        if self.renderer is not None and not self.renderer.is_running():
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'steps': self.steps,
            'centroid_position': np.mean([p.pos for p in self.creature.phys], axis=0).tolist(),
            'total_energy': self._calculate_energy()
        }
    
    def _calculate_energy(self) -> float:
        """计算系统的总能量"""
        # 计算动能
        kinetic_energy = 0.5 * np.sum([p.m * np.linalg.norm(p.v) ** 2 for p in self.creature.phys])
        
        # 计算势能
        potential_energy = np.sum([p.m * self.g * (p.pos[1] - self.ground) for p in self.creature.phys])
        
        return kinetic_energy + potential_energy
    
    def get_action_space(self) -> Dict[str, Any]:
        """获取动作空间信息"""
        num_muscles = len(self.creature.muscles)
        return {
            'shape': (num_muscles,),
            'type': 'continuous',
            'low': -1.0,
            'high': 1.0
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        """获取观察空间信息"""
        # 获取观察的维度
        obs_dim = len(self._get_observation())
        return {
            'shape': (obs_dim,),
            'type': 'continuous',
            'low': -np.inf,
            'high': np.inf
        }

# 预定义的环境创建函数

def make_env(env_id: str, **kwargs) -> PhysicsEnv:
    """
    创建预定义的环境
    
    :param env_id: 环境ID，如'Balance-v0', 'Box-v0'
    :param kwargs: 额外参数
    :return: PhysicsEnv实例
    """
    from optimized_walker import create_balance_creature, create_box_creature
    
    env_id = env_id.lower()
    
    if env_id == 'balance-v0':
        # 创建平衡环境
        creature = create_balance_creature()
        return PhysicsEnv(creature, **kwargs)
    elif env_id == 'box-v0':
        # 创建盒子环境
        creature = create_box_creature()
        return PhysicsEnv(creature, **kwargs)
    else:
        raise ValueError(f"Unknown environment ID: {env_id}")

# 为了兼容旧版本的API，提供一个与原Environment类似的类

class Environment(PhysicsEnv):
    """兼容旧版本的环境类"""
    def __init__(self, creaturelist, in3d=False, g=100, dampk=0, groundhigh=0, 
                 groundk=1000, grounddamp=100, friction=100, randsigma=0.1):
        """兼容旧版本的初始化方法"""
        # 假设creaturelist只有一个生物
        creature = creaturelist[0] if creaturelist else None
        super().__init__(creature, in3d, g, dampk, groundhigh, groundk, grounddamp, friction, randsigma)
        self.creatures = creaturelist
    
    def run(self):
        """兼容旧版本的run方法"""
        for c in self.creatures:
            c.run()
            for p in c.phys:
                p.forced([0, -self.g, 0] if self.in3d else [0, -self.g])
                self._damp(p, self.dampk)
                
                if p.pos[1] - self.ground < 0:
                    p.color = "red"
                    p.r = 3
                    deep = p.pos[1] - self.ground
                    p.forced([0, -self.ground_k * deep, 0] if self.in3d else [0, -self.ground_k * deep])
                    p.forced([0, -self.ground_damp * p.v[1], 0] if self.in3d else [0, -self.ground_damp * p.v[1]])
                    friction_force = np.abs(deep) * self.friction
                    if self.in3d:
                        p.forced([-p.v[0] * friction_force, 0, -p.v[2] * friction_force])
                    else:
                        p.forced([-p.v[0] * friction_force, 0])
                else:
                    p.color = "black"
                    p.r = 1
    
    def step(self, t):
        """兼容旧版本的step方法"""
        self.run()
        Point.run1(t)

# 确保导入pygame在需要时才进行

# 只有在实际需要渲染时才导入pygame
import pygame