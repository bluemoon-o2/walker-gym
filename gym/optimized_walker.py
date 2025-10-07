from optimized_engine import Point, DingPoint
from optimized_renderer import Renderer
import numpy as np
import time
from typing import List, Tuple, Optional

class Muscle:
    def __init__(self, p1: Point, p2: Point, x: float = None, k: float = 1000, 
                 maxl: float = 1.5, minl: float = 0.1, stride: float = 2, dampk: float = 20):
        """
        肌肉类 - 可主动收缩和舒张的弹簧
        """
        self.p1 = p1
        self.p2 = p2
        self.x = self.distant(p1, p2) if x is None else x
        self.originx = self.x
        self.k = k
        self.dampk = dampk
        self.minl = minl
        self.maxl = maxl
        self.stride = stride
    
    def distant(self, p1: Point, p2: Point) -> float:
        """计算两点之间的距离"""
        return np.linalg.norm(p1.pos - p2.pos)
    
    def regulation(self) -> None:
        """限制肌肉长度在合理范围内"""
        self.x = max(self.x, self.originx * self.minl)
        self.x = min(self.x, self.originx * self.maxl)
    
    def act(self, a: float) -> None:
        """主动收缩或舒张肌肉"""
        self.x += a
        self.regulation()
    
    def actdisp(self, a: bool) -> None:
        """离散化的肌肉动作"""
        if a:
            self.x += self.stride
        else:
            self.x -= self.stride
        self.regulation()
    
    def run(self) -> None:
        """运行肌肉的物理模拟"""
        # 计算弹力
        current_dist = self.distant(self.p1, self.p2)
        dx = current_dist - self.x
        f_size = -dx * self.k
        
        # 计算方向向量
        direction = self.p2.pos - self.p1.pos
        if current_dist > 0:
            direction = direction / current_dist
        
        # 应用弹力
        force = f_size * direction
        self.p1.forced(force)
        self.p2.forced(-force)
        
        # 计算阻尼力
        dv = self.p1.v - self.p2.v
        dk = np.dot(dv, direction)
        damp_force = dk * self.dampk * direction
        self.p1.forced(-damp_force)
        self.p2.forced(damp_force)

class Skeleton:
    def __init__(self, p1: Point, p2: Point, x: float = None, k: float = 1000, dampk: float = 20):
        """
        骨骼类 - 不可主动收缩的弹簧
        """
        self.p1 = p1
        self.p2 = p2
        self.x = self.distant(p1, p2) if x is None else x
        self.k = k
        self.dampk = dampk
    
    def distant(self, p1: Point, p2: Point) -> float:
        """计算两点之间的距离"""
        return np.linalg.norm(p1.pos - p2.pos)
    
    def run(self) -> None:
        """运行骨骼的物理模拟"""
        # 计算弹力
        current_dist = self.distant(self.p1, self.p2)
        dx = current_dist - self.x
        f_size = -dx * self.k
        
        # 计算方向向量
        direction = self.p2.pos - self.p1.pos
        if current_dist > 0:
            direction = direction / current_dist
        
        # 应用弹力
        force = f_size * direction
        self.p1.forced(force)
        self.p2.forced(-force)
        
        # 计算阻尼力
        dv = self.p1.v - self.p2.v
        dk = np.dot(dv, direction)
        damp_force = dk * self.dampk * direction
        self.p1.forced(-damp_force)
        self.p2.forced(damp_force)

class Creature:
    def __init__(self, phylist: List[Point], musclelist: List[Muscle], skeletonlist: List[Skeleton]):
        """
        生物类 - 由点、肌肉和骨骼组成的物理系统
        """
        self.phys = phylist
        self.muscles = musclelist
        self.skeletons = skeletonlist
    
    def run(self) -> None:
        """运行生物的物理模拟"""
        # 先重置所有点的加速度
        for p in self.phys:
            p.zero()
        
        # 运行肌肉和骨骼的物理模拟
        for muscle in self.muscles:
            muscle.run()
        for skeleton in self.skeletons:
            skeleton.run()
    
    def getstat(self, in3d: bool = True, pk: float = 1, vk: float = 1, ak: float = 1, 
                mk: float = 1, midform: bool = True, conmid: bool = False) -> List[float]:
        """
        获取生物的状态信息，用于强化学习等应用
        """
        s = []
        d = 3 if in3d else 2
        mid = np.zeros(3, dtype=np.float32)
        
        # 计算质心
        if midform:
            for i in self.phys:
                mid += i.pos
            mid /= len(self.phys)
        
        # 获取每个点的位置、速度和加速度信息
        for i in self.phys:
            # 位置相对于质心（如果启用）
            pos = (i.pos[:d] - mid[:d]) * pk if midform else i.pos[:d] * pk
            s.extend(pos.tolist())
            # 速度
            s.extend((i.v[:d] * vk).tolist())
            # 加速度
            s.extend((i.old_a[:d] * ak).tolist())
        
        # 是否包含质心位置
        if conmid:
            s.extend(mid.tolist())
        
        # 获取肌肉长度信息
        for i in self.muscles:
            s.append(i.x * mk)
        
        return s
    
    def act(self, a: List[float]) -> None:
        """控制生物的动作"""
        for i in range(min(len(self.muscles), len(a))):
            self.muscles[i].act(a[i])
    
    def actdisp(self, a: List[bool]) -> None:
        """控制生物的离散动作"""
        for i in range(min(len(self.muscles), len(a))):
            self.muscles[i].actdisp(a[i])

# 创建一些示例生物

def create_balance_creature() -> Creature:
    """创建一个简单的平衡生物"""
    # 创建点
    p = [
        Point(5, [-50, 100, 0], [0, 0, 0]),  # 左上方点
        Point(5, [50, 100, 0], [0, 0, 0]),   # 右上方点
        Point(1, [0, 0, 0], [0, 0, 0]),      # 中心点
        Point(3, [0, 100, 0], [0, 0, 0])     # 上方中心点
    ]
    
    # 创建骨骼
    sk = [
        Skeleton(p[0], p[1]),  # 左右连接
        Skeleton(p[0], p[3]),  # 左上到上方中心
        Skeleton(p[1], p[3])   # 右上到上方中心
    ]
    
    # 创建肌肉
    m = [
        Muscle(p[0], p[2]),    # 左下到中心的肌肉
        Muscle(p[1], p[2])     # 右下到中心的肌肉
    ]
    
    return Creature(p, m, sk)

def create_box_creature() -> Creature:
    """创建一个盒子形状的生物"""
    # 创建点
    p = [
        Point(1, [-50, 0, 0], [0, 0, 0]),    # 左下
        Point(1, [-50, 100, 0], [0, 0, 0]),  # 左上
        Point(1, [50, 100, 0], [0, 0, 0]),   # 右上
        Point(1, [50, 0, 0], [0, 0, 0])      # 右下
    ]
    
    # 创建骨骼
    sk = [
        Skeleton(p[1], p[2])   # 顶部边
    ]
    
    # 创建肌肉
    m = [
        Muscle(p[0], p[1]),    # 左侧肌肉
        Muscle(p[0], p[2]),    # 对角线肌肉
        Muscle(p[3], p[1]),    # 右上肌肉
        Muscle(p[3], p[2])     # 右侧肌肉
    ]
    
    return Creature(p, m, sk)

# 性能比较示例

def performance_comparison():
    """比较优化前后的性能差异"""
    print("开始性能比较...")
    
    # 清除所有点
    Point.clear()
    
    # 创建多个点进行性能测试
    num_points = 100
    points = []
    
    print(f"创建 {num_points} 个点...")
    for i in range(num_points):
        # 随机位置和速度
        pos = np.random.uniform(-100, 100, 3)
        v = np.random.uniform(-10, 10, 3)
        points.append(Point(1, pos, v, color="blue"))
    
    # 创建一些弹簧连接
    springs = []
    for i in range(num_points - 1):
        springs.append(Skeleton(points[i], points[i+1], k=50))
    
    # 创建一个简单的生物
    creature = Creature(points, [], springs)
    
    # 测量计算性能
    print("测量计算性能...")
    steps = 1000
    start_time = time.time()
    
    for _ in range(steps):
        # 计算物理
        creature.run()
        # 应用重力
        Point.gravity()
        # 更新位置
        Point.run1(0.01)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"计算 {steps} 步耗时: {elapsed:.4f} 秒")
    print(f"每步平均耗时: {(elapsed/steps)*1000:.4f} 毫秒")
    print(f"每秒可处理: {steps/elapsed:.2f} 步")
    
    # 启动渲染以展示可视化性能
    print("启动渲染，展示可视化性能...")
    print("按键说明:")
    print("  ESC - 退出")
    print("  F - 显示/隐藏FPS")
    print("  A - 显示/隐藏加速度向量")
    print("  V - 显示/隐藏速度向量")
    print("  WASD/空格/左Ctrl - 移动相机")
    print("  方向键 - 旋转相机")
    print("  +/- - 缩放画面")
    
    # 创建渲染器并启动渲染
    renderer = Renderer()
    renderer.play(fps=60)
    
    print("性能测试完成。")

# 创建一个示例模块，用于try.py调用
example = type('ExampleModule', (), {'performance_comparison': performance_comparison})