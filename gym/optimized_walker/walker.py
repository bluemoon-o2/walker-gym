import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from .core import Config, Point
from .env import Environment


class Muscle:
    """肌肉类，控制两点之间的收缩和伸展"""
    def __init__(self, point1: Point, point2: Point, amp: float = 1.0, freq: float = 1.0,
                 phase: float = 0.0, power: float = 100.0, x: float = None):
        """
        初始化肌肉
        
        :param point1: 连接的第一个点
        :param point2: 连接的第二个点
        :param amp: 振幅（收缩/伸展的比例）
        :param freq: 频率（每秒收缩的次数）
        :param phase: 相位（初始相位）
        :param power: 肌肉力量（劲度系数）
        :param x: 肌肉原长
        """
        self.point1 = point1
        self.point2 = point2
        self.amp = amp
        self.freq = freq
        self.phase = phase
        self.power = power
        
        # 计算肌肉原长
        if x is None:
            self.x = np.linalg.norm(point1.pos - point2.pos).astype(Config.precision)
        else:
            self.x = x
            
        # 肌肉状态
        self.t = 0  # 当前时间
        self.state = 0  # 肌肉当前状态（0-1）
        self.active = True  # 是否激活

    def __repr__(self):
        return f"Muscle(amp={self.amp}, freq={self.freq}, phase={self.phase}, power={self.power})"

    def params(self) -> dict:
        """返回肌肉的参数字典"""
        return {
            "amp": self.amp,
            "freq": self.freq,
            "phase": self.phase,
            "power": self.power,
            "x": self.x,
            "t": self.t,
            "state": self.state,
            "active": self.active
        }

    def act(self, dt: float) -> float:
        """
        肌肉收缩/伸展的动作
        
        :param dt: 时间步长
        :return: 肌肉当前的收缩状态（0-1）
        """
        if not self.active:
            return self.state
            
        # 更新时间
        self.t += dt
        
        # 计算肌肉收缩状态（使用正弦函数）
        self.state = (np.sin(2 * np.pi * self.freq * self.t + self.phase) + 1) / 2
        
        # 应用肌肉力量（改变弹簧长度）
        target_length = self.x * (1 - self.amp * self.state)
        
        # 计算当前长度
        current_length = np.linalg.norm(self.point1.pos - self.point2.pos).astype(Config.precision)
        
        # 计算力的大小
        force_magnitude = (target_length - current_length) * self.power
        
        # 应用力到两个点
        direction = self.point2.pos - self.point1.pos
        if np.linalg.norm(direction).astype(Config.precision) > Config.r:
            direction = direction / np.linalg.norm(direction).astype(Config.precision)
            
            force_vector = force_magnitude * direction
            self.point1.forced(force_vector)
            self.point2.forced(-force_vector)
            
        return self.state

    def actdisp(self, dt: float, disp: float) -> float:
        """
        根据位移控制肌肉动作
        
        :param dt: 时间步长
        :param disp: 位移信号
        :return: 肌肉当前的收缩状态（0-1）
        """
        if not self.active:
            return self.state
            
        # 更新时间
        self.t += dt
        
        # 使用位移信号控制肌肉状态
        self.state = np.clip(disp, 0, 1)
        
        # 应用肌肉力量（改变弹簧长度）
        target_length = self.x * (1 - self.amp * self.state)
        
        # 计算当前长度
        current_length = np.linalg.norm(self.point1.pos - self.point2.pos).astype(Config.precision)
        
        # 计算力的大小
        force_magnitude = (target_length - current_length) * self.power
        
        # 应用力到两个点
        direction = self.point2.pos - self.point1.pos
        if np.linalg.norm(direction).astype(Config.precision) > Config.r:
            direction = direction / np.linalg.norm(direction).astype(Config.precision)
            
            force_vector = force_magnitude * direction
            self.point1.forced(force_vector)
            self.point2.forced(-force_vector)
            
        return self.state

    def run(self, dt: float) -> None:
        """运行肌肉模拟"""
        self.act(dt)

    def toggle(self) -> None:
        """切换肌肉激活状态"""
        self.active = not self.active

    def set_params(self, **kwargs) -> None:
        """设置肌肉参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Skeleton:
    """骨骼类，定义生物的骨架结构"""
    def __init__(self, env: Environment):
        """
        初始化骨骼
        
        :param env: 物理环境实例
        """
        self.env = env
        self.points = []        # 骨骼点列表
        self.springs = []       # 骨骼连接的弹簧列表
        self.muscles = []       # 肌肉列表
        
    def add_point(self, m: float, pos: Union[np.ndarray, tuple, list],
                 v: Union[np.ndarray, tuple, list] = (0, 0, 0), r: float = None,
                 color: Union[str, Tuple[int, int, int]] = "black", is_ding: bool = False) -> Point:
        """
        添加骨骼点
        
        :param m: 质量
        :param pos: 位置
        :param v: 速度
        :param r: 半径
        :param color: 颜色
        :param is_ding: 是否为定点
        :return: 创建的点对象
        """
        if is_ding:
            point = self.env.add_ding_point(m, pos, v, r, color)
        else:
            point = self.env.add_point(m, pos, v, r, color)
            
        self.points.append(point)
        return point
        
    def add_spring(self, point1: Point, point2: Point, k: float = 100, x: float = None,
                   string: bool = False) -> None:
        """
        添加骨骼连接（弹簧）
        
        :param point1: 第一个点
        :param point2: 第二个点
        :param k: 劲度系数
        :param x: 弹簧原长
        :param string: 是否为绳型弹簧
        """
        self.env.add_spring(point1, point2, x, k, string)
        self.springs.append((point1, point2))
        
    def add_muscle(self, point1: Point, point2: Point, amp: float = 1.0, freq: float = 1.0,
                   phase: float = 0.0, power: float = 100.0, x: float = None) -> Muscle:
        """
        添加肌肉
        
        :param point1: 第一个连接点
        :param point2: 第二个连接点
        :param amp: 振幅
        :param freq: 频率
        :param phase: 相位
        :param power: 肌肉力量
        :param x: 肌肉原长
        :return: 创建的肌肉对象
        """
        muscle = Muscle(point1, point2, amp, freq, phase, power, x)
        self.muscles.append(muscle)
        return muscle
        
    def update(self, dt: float) -> None:
        """
        更新骨骼状态
        
        :param dt: 时间步长
        """
        # 更新所有肌肉
        for muscle in self.muscles:
            muscle.run(dt)

class Creature:
    """生物类，组合骨骼和肌肉，控制生物的运动"""
    def __init__(self, env: Environment, skeleton: Optional[Skeleton] = None):
        """
        初始化生物
        
        :param env: 物理环境实例
        :param skeleton: 骨骼实例，如果为None则创建新的骨骼
        """
        self.env = env
        
        if skeleton is None:
            self.skeleton = Skeleton(env)
        else:
            self.skeleton = skeleton
            
        # 生物状态
        self.brain = None      # 生物的控制系统
        self.fitness = 0.0     # 适应度
        self.age = 0           # 年龄（步数）
        
    def __repr__(self):
        return f"Creature(fitness={self.fitness}, age={self.age})"

    def act(self, dt: float) -> None:
        """
        生物动作
        
        :param dt: 时间步长
        """
        # 更新骨骼和肌肉
        self.skeleton.update(dt)
        
        # 如果有大脑，使用大脑控制肌肉
        if self.brain is not None:
            self.brain.control(self.skeleton.muscles, dt)
            
        # 增加年龄
        self.age += 1

    def actdisp(self, dt: float, disp: List[float]) -> None:
        """
        使用位移信号控制生物动作
        
        :param dt: 时间步长
        :param disp: 位移信号列表
        """
        # 确保位移信号数量与肌肉数量匹配
        if len(disp) < len(self.skeleton.muscles):
            # 扩展位移信号列表
            disp = disp + [0.0] * (len(self.skeleton.muscles) - len(disp))
        elif len(disp) > len(self.skeleton.muscles):
            # 截断位移信号列表
            disp = disp[:len(self.skeleton.muscles)]
            
        # 使用位移信号控制肌肉
        for i, muscle in enumerate(self.skeleton.muscles):
            muscle.actdisp(dt, disp[i])
            
        # 增加年龄
        self.age += 1

    def evaluate_fitness(self) -> float:
        """
        评估生物的适应度
        
        :return: 适应度值
        """
        # 默认适应度计算：质心的x坐标（向前移动的距离）
        if not self.skeleton.points:
            return 0.0
            
        # 计算质心
        total_mass = 0.0
        center_of_mass = np.zeros(3, dtype=Config.precision)
        
        for point in self.skeleton.points:
            total_mass += point.m
            center_of_mass += point.pos * point.m
            
        if total_mass > 0:
            center_of_mass /= total_mass
        
        # 设置适应度为质心的x坐标
        self.fitness = center_of_mass[0]
        
        return self.fitness

    def set_brain(self, brain: Callable) -> None:
        """
        设置生物的控制系统
        
        :param brain: 大脑函数，接受肌肉列表和时间步长作为参数
        """
        self.brain = brain


class Brain:
    """简单的生物控制系统"""
    def __init__(self, pattern: List[Dict] = None):
        """
        初始化大脑
        
        :param pattern: 控制模式列表，每个元素是包含肌肉控制参数的字典
        """
        self.pattern = pattern or []
        self.t = 0

    def control(self, muscles: List[Muscle], dt: float) -> None:
        """
        控制肌肉
        
        :param muscles: 肌肉列表
        :param dt: 时间步长
        """
        # 更新时间
        self.t += dt
        
        # 如果有控制模式，应用控制模式
        if self.pattern and len(self.pattern) >= len(muscles):
            for i, muscle in enumerate(muscles):
                pattern = self.pattern[i]
                
                # 根据模式控制肌肉
                if 'amp' in pattern:
                    muscle.amp = pattern['amp']
                if 'freq' in pattern:
                    muscle.freq = pattern['freq']
                if 'phase' in pattern:
                    muscle.phase = pattern['phase']
                if 'power' in pattern:
                    muscle.power = pattern['power']

# 生物结构定义函数

def test(env: Environment) -> Creature:
    """创建一个简单的测试生物"""
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加点
    p1 = skeleton.add_point(1, (0, 0, 0))
    p2 = skeleton.add_point(1, (10, 0, 0))
    
    # 添加弹簧连接
    skeleton.add_spring(p1, p2)
    
    # 添加肌肉
    skeleton.add_muscle(p1, p2, amp=0.1, freq=1)
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    return creature


def leg2(env: Environment) -> Creature:
    """创建一个双足生物"""
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加身体点
    body = skeleton.add_point(5, (0, 10, 0), r=3)
    
    # 添加腿部点
    leg1_hip = skeleton.add_point(1, (-5, 5, 0))
    leg1_knee = skeleton.add_point(1, (-5, -5, 0))
    leg1_foot = skeleton.add_point(2, (-5, -15, 0), r=2)
    
    leg2_hip = skeleton.add_point(1, (5, 5, 0))
    leg2_knee = skeleton.add_point(1, (5, -5, 0))
    leg2_foot = skeleton.add_point(2, (5, -15, 0), r=2)
    
    # 添加骨骼连接
    skeleton.add_spring(body, leg1_hip, k=500)
    skeleton.add_spring(leg1_hip, leg1_knee, k=300)
    skeleton.add_spring(leg1_knee, leg1_foot, k=300)
    
    skeleton.add_spring(body, leg2_hip, k=500)
    skeleton.add_spring(leg2_hip, leg2_knee, k=300)
    skeleton.add_spring(leg2_knee, leg2_foot, k=300)
    
    # 添加肌肉
    # 腿部肌肉（控制膝盖弯曲）
    skeleton.add_muscle(leg1_hip, leg1_knee, amp=0.1, freq=0.5, phase=0, power=200)
    skeleton.add_muscle(leg1_knee, leg1_foot, amp=0.1, freq=0.5, phase=0.5, power=200)
    
    skeleton.add_muscle(leg2_hip, leg2_knee, amp=0.1, freq=0.5, phase=0.5, power=200)
    skeleton.add_muscle(leg2_knee, leg2_foot, amp=0.1, freq=0.5, phase=0, power=200)
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    return creature


def box(env: Environment, size: float = 10, mass: float = 1) -> Creature:
    """创建一个立方体生物"""
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加立方体的8个顶点
    p1 = skeleton.add_point(mass, (-size/2, size/2, -size/2))
    p2 = skeleton.add_point(mass, (size/2, size/2, -size/2))
    p3 = skeleton.add_point(mass, (size/2, -size/2, -size/2))
    p4 = skeleton.add_point(mass, (-size/2, -size/2, -size/2))
    p5 = skeleton.add_point(mass, (-size/2, size/2, size/2))
    p6 = skeleton.add_point(mass, (size/2, size/2, size/2))
    p7 = skeleton.add_point(mass, (size/2, -size/2, size/2))
    p8 = skeleton.add_point(mass, (-size/2, -size/2, size/2))
    
    # 添加立方体的12条边（弹簧连接）
    skeleton.add_spring(p1, p2, k=500)
    skeleton.add_spring(p2, p3, k=500)
    skeleton.add_spring(p3, p4, k=500)
    skeleton.add_spring(p4, p1, k=500)
    
    skeleton.add_spring(p5, p6, k=500)
    skeleton.add_spring(p6, p7, k=500)
    skeleton.add_spring(p7, p8, k=500)
    skeleton.add_spring(p8, p5, k=500)
    
    skeleton.add_spring(p1, p5, k=500)
    skeleton.add_spring(p2, p6, k=500)
    skeleton.add_spring(p3, p7, k=500)
    skeleton.add_spring(p4, p8, k=500)
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    return creature


def balance1(env: Environment) -> Creature:
    """创建一个单摆平衡生物"""
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加定点（悬挂点）
    pivot = skeleton.add_point(0, (0, 20, 0), is_ding=True, color="red")
    
    # 添加摆锤点
    pendulum = skeleton.add_point(5, (0, 0, 0), r=3)
    
    # 添加弹簧连接
    skeleton.add_spring(pivot, pendulum, k=200)
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    return creature


def balance2(env: Environment) -> Creature:
    """创建一个双摆平衡生物"""
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加定点（悬挂点）
    pivot = skeleton.add_point(0, (0, 20, 0), is_ding=True, color="red")
    
    # 添加摆锤点
    pendulum1 = skeleton.add_point(2, (0, 10, 0))
    pendulum2 = skeleton.add_point(2, (0, 0, 0), r=2)
    
    # 添加弹簧连接
    skeleton.add_spring(pivot, pendulum1, k=200)
    skeleton.add_spring(pendulum1, pendulum2, k=200)
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    return creature


def balance3(env: Environment) -> Creature:
    """创建一个三摆平衡生物"""
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加定点（悬挂点）
    pivot = skeleton.add_point(0, (0, 20, 0), is_ding=True, color="red")
    
    # 添加摆锤点
    pendulum1 = skeleton.add_point(1.5, (0, 15, 0))
    pendulum2 = skeleton.add_point(1.5, (0, 10, 0))
    pendulum3 = skeleton.add_point(1.5, (0, 0, 0), r=2)
    
    # 添加弹簧连接
    skeleton.add_spring(pivot, pendulum1, k=200)
    skeleton.add_spring(pendulum1, pendulum2, k=200)
    skeleton.add_spring(pendulum2, pendulum3, k=200)
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    return creature


def humanb(env: Environment) -> Creature:
    """创建一个类人型生物"""
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加身体点
    head = skeleton.add_point(3, (0, 30, 0), r=3, color="blue")
    torso = skeleton.add_point(10, (0, 20, 0), r=4)
    
    # 添加手臂点
    left_shoulder = skeleton.add_point(2, (-8, 25, 0))
    left_elbow = skeleton.add_point(1, (-15, 20, 0))
    left_hand = skeleton.add_point(1, (-20, 20, 0))
    
    right_shoulder = skeleton.add_point(2, (8, 25, 0))
    right_elbow = skeleton.add_point(1, (15, 20, 0))
    right_hand = skeleton.add_point(1, (20, 20, 0))
    
    # 添加腿部点
    left_hip = skeleton.add_point(2, (-5, 10, 0))
    left_knee = skeleton.add_point(1, (-5, 0, 0))
    left_foot = skeleton.add_point(2, (-5, -10, 0), r=2)
    
    right_hip = skeleton.add_point(2, (5, 10, 0))
    right_knee = skeleton.add_point(1, (5, 0, 0))
    right_foot = skeleton.add_point(2, (5, -10, 0), r=2)
    
    # 添加骨骼连接
    skeleton.add_spring(head, torso, k=500)
    
    skeleton.add_spring(torso, left_shoulder, k=400)
    skeleton.add_spring(left_shoulder, left_elbow, k=300)
    skeleton.add_spring(left_elbow, left_hand, k=200)
    
    skeleton.add_spring(torso, right_shoulder, k=400)
    skeleton.add_spring(right_shoulder, right_elbow, k=300)
    skeleton.add_spring(right_elbow, right_hand, k=200)
    
    skeleton.add_spring(torso, left_hip, k=500)
    skeleton.add_spring(left_hip, left_knee, k=400)
    skeleton.add_spring(left_knee, left_foot, k=400)
    
    skeleton.add_spring(torso, right_hip, k=500)
    skeleton.add_spring(right_hip, right_knee, k=400)
    skeleton.add_spring(right_knee, right_foot, k=400)
    
    # 添加肌肉
    # 手臂肌肉
    skeleton.add_muscle(torso, left_elbow, amp=0.1, freq=0.3, phase=0, power=150)
    skeleton.add_muscle(left_shoulder, left_hand, amp=0.1, freq=0.3, phase=0.5, power=100)
    
    skeleton.add_muscle(torso, right_elbow, amp=0.1, freq=0.3, phase=0.5, power=150)
    skeleton.add_muscle(right_shoulder, right_hand, amp=0.1, freq=0.3, phase=0, power=100)
    
    # 腿部肌肉
    skeleton.add_muscle(torso, left_knee, amp=0.1, freq=0.5, phase=0, power=200)
    skeleton.add_muscle(left_hip, left_foot, amp=0.1, freq=0.5, phase=0.5, power=150)
    
    skeleton.add_muscle(torso, right_knee, amp=0.1, freq=0.5, phase=0.5, power=200)
    skeleton.add_muscle(right_hip, right_foot, amp=0.1, freq=0.5, phase=0, power=150)
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    return creature


def insect(env: Environment, legs: int = 6) -> Creature:
    """创建一个昆虫型生物"""
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加身体点
    body_length = legs * 5  # 根据腿的数量调整身体长度
    body_points = []
    for i in range(legs // 2):
        x = -body_length / 2 + i * (body_length / (legs // 2 - 1)) if legs > 2 else 0
        body_points.append(skeleton.add_point(2, (x, 5, 0), r=2))
    
    # 连接身体点
    for i in range(len(body_points) - 1):
        skeleton.add_spring(body_points[i], body_points[i+1], k=400)
    
    # 添加腿部
    leg_pairs = []
    for i, body_point in enumerate(body_points):
        # 左腿
        left_upper = skeleton.add_point(1, (body_point.pos[0] - 5, 0, 0))
        left_lower = skeleton.add_point(1, (body_point.pos[0] - 10, -5, 0))
        left_foot = skeleton.add_point(1, (body_point.pos[0] - 15, -10, 0), r=1.5)
        
        # 右腿
        right_upper = skeleton.add_point(1, (body_point.pos[0] + 5, 0, 0))
        right_lower = skeleton.add_point(1, (body_point.pos[0] + 10, -5, 0))
        right_foot = skeleton.add_point(1, (body_point.pos[0] + 15, -10, 0), r=1.5)
        
        # 添加腿部连接
        skeleton.add_spring(body_point, left_upper, k=300)
        skeleton.add_spring(left_upper, left_lower, k=200)
        skeleton.add_spring(left_lower, left_foot, k=200)
        
        skeleton.add_spring(body_point, right_upper, k=300)
        skeleton.add_spring(right_upper, right_lower, k=200)
        skeleton.add_spring(right_lower, right_foot, k=200)
        
        # 添加腿部肌肉
        phase = i * (np.pi / (legs // 2))  # 不同腿的相位不同，形成行走步态
        
        skeleton.add_muscle(body_point, left_lower, amp=0.1, freq=0.8, phase=phase, power=100)
        skeleton.add_muscle(left_upper, left_foot, amp=0.1, freq=0.8, phase=phase + 0.5, power=80)
        
        skeleton.add_muscle(body_point, right_lower, amp=0.1, freq=0.8, phase=phase + np.pi, power=100)
        skeleton.add_muscle(right_upper, right_foot, amp=0.1, freq=0.8, phase=phase + np.pi + 0.5, power=80)
        
        leg_pairs.append((left_foot, right_foot))
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    return creature

# 导出常用类和函数
export = {
    "Muscle": Muscle,
    "Skeleton": Skeleton,
    "Creature": Creature,
    "Brain": Brain,
    "test": test,
    "leg2": leg2,
    "box": box,
    "balance1": balance1,
    "balance2": balance2,
    "balance3": balance3,
    "humanb": humanb,
    "insect": insect
}