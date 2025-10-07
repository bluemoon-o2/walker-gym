import numpy as np
from typing import Union, Tuple, List, Dict, Optional
import time

class Config:
    precision = np.float32
    r = 16e-36
    e = 16e-20
    k = 8.99e9
    g = 9.8

class Point:
    """优化后的点对象"""
    # 使用类变量存储所有点的信息，便于向量化操作
    points = []   # 点对象列表
    r_points = {} # 储存需要连成弹簧的点
    fps = 0       # 帧数记录

    # 类变量，用于存储所有点的状态，便于向量化计算
    positions = np.array([])  # 所有点的位置 (n, 3)
    velocities = np.array([]) # 所有点的速度 (n, 3)
    accelerations = np.array([]) # 所有点的加速度 (n, 3)
    masses = np.array([])     # 所有点的质量 (n,)
    radii = np.array([])      # 所有点的半径 (n,)
    charges = np.array([])    # 所有点的电荷 (n,)
    colors = []               # 所有点的颜色

    @classmethod
    def clear(cls):
        """清空所有点"""
        cls.points = []
        cls.r_points = {}
        cls.fps = 0
        cls.positions = np.array([])
        cls.velocities = np.array([])
        cls.accelerations = np.array([])
        cls.masses = np.array([])
        cls.radii = np.array([])
        cls.charges = np.array([])
        cls.colors = []

    def __init__(self, m: float, pos: Union[np.ndarray, tuple, list],
                 v: Union[np.ndarray, tuple, list], r: float = None,
                 color: Union[str, Tuple[int, int, int]]="black", e: float = Config.e):
        """
        :param m: 质量
        :param pos: 位置
        :param v: 速度
        :param color: 点的颜色
        :param r: 点的半径
        :param e: 电荷
        """
        self.m = m
        self.pos = np.array(pos, dtype=Config.precision)
        self.v = np.array(v, dtype=Config.precision)
        self.a = np.zeros_like(self.v, dtype=Config.precision)
        if r is None:
            r = m ** 0.3
        self.r = r
        self.old_a = self.a.copy()
        self.color = color
        self.e = e
        
        # 将点添加到类变量中
        Point.points.append(self)
        
        # 更新类数组
        self._update_class_arrays()
    
    def _update_class_arrays(self):
        """更新类数组，保持所有点的信息同步"""
        # 获取当前点的索引
        idx = len(Point.points) - 1
        
        # 初始化或扩展数组
        if idx == 0:
            Point.positions = np.array([self.pos], dtype=Config.precision)
            Point.velocities = np.array([self.v], dtype=Config.precision)
            Point.accelerations = np.array([self.a], dtype=Config.precision)
            Point.masses = np.array([self.m], dtype=Config.precision)
            Point.radii = np.array([self.r], dtype=Config.precision)
            Point.charges = np.array([self.e], dtype=Config.precision)
        else:
            Point.positions = np.append(Point.positions, [self.pos], axis=0)
            Point.velocities = np.append(Point.velocities, [self.v], axis=0)
            Point.accelerations = np.append(Point.accelerations, [self.a], axis=0)
            Point.masses = np.append(Point.masses, [self.m])
            Point.radii = np.append(Point.radii, [self.r])
            Point.charges = np.append(Point.charges, [self.e])
        
        Point.colors.append(self.color)
    
    def __repr__(self):
        return f"Point(m={self.m}, pos={self.pos}, v={self.v}, a={self.old_a})"
    
    def params(self):
        return {"m": self.m, "v": self.v.tolist(), "a": self.a.tolist(), "pos": self.pos.tolist(),
                "r": self.r, "e": self.e, "color": self.color, "old_a": self.old_a.tolist()}
    
    def zero(self) -> None:
        """将加速度设为0"""
        self.a = np.zeros_like(self.v, dtype=Config.precision)
    
    def forced(self, f: np.ndarray) -> None:
        """受力"""
        self.a += f / self.m
    
    def anti_forced(self, f_size: float, target: 'Point') -> None:
        """受反作用力"""
        direction = target.pos - self.pos
        distance = np.linalg.norm(direction).astype(float)
        distance = max(distance, Config.r)
        force = -f_size * direction / distance
        self.forced(force)
    
    def resilience(self, other: 'Point', x: float = None, k: float = 100, string: bool = False) -> None:
        """
        对另一点施弹力
        :param x: 弹簧原长，默认当前长度为原长
        :param k: 劲度系数
        :param other: 弹簧的另一个点
        :param string: 绳型（True）或杆型（False）
        """
        current = np.linalg.norm(self.pos - other.pos)
        key = tuple(sorted([self, other], key=id))
        if x is None:
            if key not in Point.r_points:
                x = current
                Point.r_points[key] = x
            else:
                x = Point.r_points[key]
        else:
            Point.r_points[key] = x
        dx = current - x
        if dx < 0 and string:
            f_size = 0
        else:
            f_size = -dx * k
        self.anti_forced(f_size, other)
        other.anti_forced(f_size, self)
    
    @classmethod
    def all_resilience(cls, r_list: List[dict]) -> None:
        """
        批量施弹力
        :param r_list: [{"self":Point, "other":Point, "x":float, "k":float, "string":bool},...]
        :return: None
        """
        for i in r_list:
            i["self"].resilience(i["other"], i["x"], i["k"], i["string"])
    
    def bounce(self, k: float = 100, other: Union[str, List['Point']] = "*") -> None:
        """
        弹簧模拟碰撞
        :param k: 劲度系数
        :param other: 被碰撞的一组物体，当为"*"时指对所有点
        """
        if other == "*":
            other = Point.points
        for i in other:
            if i == self:
                continue
            elif np.linalg.norm(self.pos - i.pos).astype(float) <= self.r + i.r:
                self.resilience(i, self.r + i.r, k / 2)
    
    @classmethod
    def gravity_vec(cls) -> None:
        """使用向量化操作计算全局引力 - 优化版本"""
        n = len(cls.points)
        if n < 2:
            return
        
        # 重置所有加速度
        for p in cls.points:
            p.zero()
        
        # 向量化计算引力
        # 计算所有点对之间的引力
        for i in range(n):
            for j in range(i + 1, n):
                p1 = cls.points[i]
                p2 = cls.points[j]
                
                # 计算距离向量和距离
                direction = p2.pos - p1.pos
                distance = np.linalg.norm(direction)
                distance = max(distance, Config.r)
                
                # 计算引力大小
                f = -Config.g * p1.m * p2.m / (distance ** 2)
                
                # 计算引力向量并应用
                force = f * direction / distance
                p1.forced(force)
                p2.forced(-force)
    
    @classmethod
    def gravity(cls) -> None:
        """全局引力 - 兼容旧版本"""
        return cls.gravity_vec()
    
    @classmethod
    def coulomb_vec(cls) -> None:
        """使用向量化操作计算全局静电力 - 优化版本"""
        n = len(cls.points)
        if n < 2:
            return
        
        # 重置所有加速度
        for p in cls.points:
            p.zero()
        
        # 向量化计算静电力
        # 计算所有点对之间的静电力
        for i in range(n):
            for j in range(i + 1, n):
                p1 = cls.points[i]
                p2 = cls.points[j]
                
                # 计算距离向量和距离
                direction = p2.pos - p1.pos
                distance = np.linalg.norm(direction)
                distance = max(distance, Config.r)
                
                # 计算静电力大小
                f = -Config.k * p1.e * p2.e / (distance ** 2)
                
                # 计算静电力向量并应用
                force = f * direction / distance
                p1.forced(force)
                p2.forced(-force)
    
    @classmethod
    def coulomb(cls) -> None:
        """全局静电力 - 兼容旧版本"""
        return cls.coulomb_vec()
    
    def electrostatic(self) -> None:
        """受集体静电力"""
        for i in Point.points:
            if i == self:
                continue
            r = np.linalg.norm(self.pos - i.pos).astype(float)
            r = max(r, Config.r)
            f = -Config.k * self.e * i.e / (r ** 2)
            self.anti_forced(f, i)
    
    @classmethod
    def momentum(cls):
        """计算全局动量和"""
        # 使用向量化操作优化
        if len(cls.points) == 0:
            return np.zeros(3, dtype=Config.precision)
        
        # 使用类数组进行向量化计算
        return np.sum(cls.velocities * cls.masses[:, np.newaxis], axis=0)
    
    @classmethod
    def run1(cls, t: float) -> None:
        """
        欧拉方法（一阶精度）- 优化版本
        :param t: 时间间隔
        """
        # 更新所有点的速度和位置
        for p in cls.points:
            p.v += p.a * t
            p.pos += p.v * t
            p.old_a = p.a.copy()
            p.zero()
        
        # 同步类数组
        cls._sync_class_arrays()
    
    @classmethod
    def run2(cls, t: float) -> None:
        """
        二阶龙格-库塔法（二阶精度）- 优化版本
        :param t: 时间间隔
        """
        # 更新所有点的位置和速度
        for p in cls.points:
            p.pos += p.v * t + 0.5 * p.a * t ** 2
            p.v += p.a * t
            p.old_a = p.a.copy()
            p.zero()
        
        # 同步类数组
        cls._sync_class_arrays()
    
    @classmethod
    def _sync_class_arrays(cls):
        """同步类数组，确保与点对象的状态一致"""
        n = len(cls.points)
        if n == 0:
            cls.positions = np.array([])
            cls.velocities = np.array([])
            cls.accelerations = np.array([])
            cls.masses = np.array([])
            cls.radii = np.array([])
            cls.charges = np.array([])
            cls.colors = []
            return
        
        # 更新类数组
        cls.positions = np.array([p.pos for p in cls.points], dtype=Config.precision)
        cls.velocities = np.array([p.v for p in cls.points], dtype=Config.precision)
        cls.accelerations = np.array([p.a for p in cls.points], dtype=Config.precision)
        cls.masses = np.array([p.m for p in cls.points])
        cls.radii = np.array([p.r for p in cls.points])
        cls.charges = np.array([p.e for p in cls.points])
        cls.colors = [p.color for p in cls.points]
    
    @classmethod
    def ready(cls) -> None:
        """初始化显示模块"""
        # 这个方法在优化版本中主要是为了兼容旧代码
        pass
    
    @classmethod
    def snapshot(cls, path="state.pkl") -> None:
        """保存快照"""
        import pickle
        state = {"points": cls.points, "r_points": cls.r_points}
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)
    
    @classmethod
    def backup(cls, path="state.pkl") -> None:
        """读取快照"""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        cls.points = state["points"]
        cls.r_points = state["r_points"]
        # 同步类数组
        cls._sync_class_arrays()
    
    @classmethod
    def perspective(cls, d: np.ndarray, cam: np.ndarray, k: float) -> np.ndarray:
        """
        透视变换
        :param d: 被变换的点
        :param cam: 相机坐标，相机朝向z轴正半轴方向
        :param k: 放大倍率
        :return: 变换后位置
        """
        t = d - cam
        if t[2] < Config.r:  # 忽略相机后方的点
            return np.zeros_like(d[:2])
        projected = t * k / t[2]
        return projected[:2]
    
    @classmethod
    def eye_z(cls, fm: np.ndarray, to: np.ndarray) -> np.ndarray:
        """x-z平面旋转，消除z分量"""
        dx = to[0] - fm[0]
        dz = to[2] - fm[2]
        distance = np.linalg.norm([dx, dz]).astype(float)
        distance = max(distance, Config.r)
        unit_x, unit_z = dx / distance, dz / distance
        return np.array([
            [unit_x, 0, unit_z],
            [0, 1, 0],
            [-unit_z, 0, unit_x]
        ])
    
    @classmethod
    def eye_y(cls, fm: np.ndarray, to: np.ndarray) -> np.ndarray:
        """x-y平面旋转，消除y分量"""
        dx = to[0] - fm[0]
        dy = to[1] - fm[1]
        distance = np.linalg.norm([dx, dy]).astype(float)
        distance = max(distance, Config.r)
        unit_x, unit_y = dx / distance, dy / distance
        return np.array([
            [unit_x, unit_y, 0],
            [-unit_y, unit_x, 0],
            [0, 0, 1]
        ])
    
    @classmethod
    def eye(cls, fm: np.ndarray, to: np.ndarray) -> np.ndarray:
        mx = cls.eye_z(fm, to)
        fm_rot = mx @ fm
        to_rot = mx @ to
        mz = cls.eye_y(fm_rot, to_rot)
        final_rot = mz @ mx
        return final_rot
    
    @classmethod
    def trans(cls, pos: np.ndarray, x: np.ndarray, c: np.ndarray = None) -> np.ndarray:
        """
        计算坐标点经过线性变换后的位置
        :param pos: 世界坐标系下的坐标
        :param c:  参考系坐标
        :param x: 线性变换矩阵
        """
        if c is None:
            c = np.zeros_like(pos, dtype=Config.precision)
        if x is None:
            x = np.eye(3, dtype=Config.precision)
        return x @ (pos - c) + c

class DingPoint(Point):
    """定点，不参与力的计算"""
    def __init__(self, m, p, v=None, r=None, color="black"):
        if v is None:
            v = [0, 0, 0]
        super().__init__(m, p, v, r, color)
        # 定点的位置不会改变
        self.original_pos = self.pos.copy()
    
    def forced(self, f: np.ndarray) -> None:
        """定点不受力"""
        pass
    
    @classmethod
    def run1(cls, t: float) -> None:
        """定点的位置不会改变"""
        for p in cls.points:
            if isinstance(p, DingPoint):
                p.pos = p.original_pos.copy()
                p.v = np.zeros_like(p.v)
        
        # 同步类数组
        cls._sync_class_arrays()
        