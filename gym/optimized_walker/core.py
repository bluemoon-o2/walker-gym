import numpy as np
from typing import Union, Tuple, List


class Config:
    """物理引擎配置参数"""
    # 使用float32以减少内存占用和提高计算效率
    precision = np.float32
    r = 16e-36  # 最小距离，避免除零错误
    e = 16e-20  # 基本电荷
    k = 8.99e9  # 库仑常数
    g = 9.8     # 重力加速度

    # 性能优化参数
    batch_size = 100  # 批量处理的大小


def to_data(data: Union[np.ndarray, tuple, list]) -> np.ndarray:
    """规整数据为numpy数组"""
    if isinstance(data, (tuple, list)):
        return np.array(data, dtype=Config.precision)
    if isinstance(data, np.ndarray):
        return data.astype(Config.precision)
    else:
        raise TypeError(f"Data must be a numpy array, tuple, list (not {type(data).__name__})")


class Point:
    """物理点对象"""
    # 使用类变量存储所有点，提高批处理效率
    points = []   # 记录所有点
    r_points = {} # 储存需要连成弹簧的点
    fps = 0       # 帧数记录

    def __init__(self, m: float, pos: Union[np.ndarray, tuple, list],
                 v: Union[np.ndarray, tuple, list], r: float = None,
                 color: Union[str, Tuple[int, int, int]]="black", e: float = Config.e):
        """
        初始化物理点
        
        :param m: 质量
        :param pos: 位置 [x, y, z]
        :param v: 速度 [vx, vy, vz]
        :param color: 点的颜色
        :param r: 点的半径
        :param e: 电荷
        """
        self.m = m
        self.pos = to_data(pos)
        self.v = to_data(v)
        self.a = np.zeros_like(self.v, dtype=Config.precision)
        if r is None:
            r = m ** 0.3
        self.r = r
        self.old_a = self.a.copy()
        self.color = color
        self.e = e
        # 使用列表而不是数组来存储加速度，提高修改效率
        Point.points.append(self)

    def __repr__(self):
        return f"Point(m={self.m}, pos={self.pos}, v={self.v}, a={self.old_a})"

    def params(self) -> dict:
        """返回点的参数字典"""
        return {
            "m": self.m,
            "v": self.v.tolist(), 
            "a": self.a.tolist(), 
            "pos": self.pos.tolist(),
            "r": self.r, 
            "e": self.e, 
            "color": self.color, 
            "old_a": self.old_a.tolist()
        }

    def zero(self) -> None:
        """将加速度设为0"""
        self.a[:] = 0.0

    def forced(self, f: np.ndarray) -> None:
        """受力"""
        self.a += f / self.m

    def anti_forced(self, f_size: float, target: 'Point') -> None:
        """受反作用力"""
        direction = target.pos - self.pos
        # 等价于sqrt(x²+y²+z²)
        distance = max(np.linalg.norm(direction).astype(Config.precision), Config.r)
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
        current = np.linalg.norm(self.pos - other.pos).astype(Config.precision)
        # 使用对象ID作为键，提高查找效率
        key = tuple(sorted([id(self), id(other)]))
        
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
        
        :param r_list: [{'self':Point, 'other':Point, 'x':float, 'k':float, 'string':bool},...]
        """
        for i in r_list:
            i['self'].resilience(i['other'], i['x'], i['k'], i['string'])

    def bounce(self, k: float = 100, other: Union[str, List['Point']] = "*") -> None:
        """
        弹簧模拟碰撞
        
        :param k: 劲度系数
        :param other: 被碰撞的一组物体，当为"*"时指对所有点
        """
        if other == "*":
            other = Point.points
            
        # 预先计算本点的位置和半径，减少重复计算
        pos_self = self.pos
        r_self = self.r
        
        for i in other:
            if i == self:
                continue
            # 只在距离足够小时计算碰撞
            if np.linalg.norm(pos_self - i.pos).astype(Config.precision) <= r_self + i.r:
                self.resilience(i, r_self + i.r, k / 2)

    @classmethod
    def gravity(cls) -> None:
        """全局引力计算，优化版使用批处理"""
        if not cls.points or len(cls.points) < 2:
            return

        for i in range(len(cls.points)):
            for j in range(i + 1, len(cls.points)):
                p1 = cls.points[i]
                p2 = cls.points[j]
                r = np.linalg.norm(p1.pos - p2.pos).astype(Config.precision)
                r = max(r, Config.r)
                f = -Config.g * p1.m * p2.m / (r ** 2)
                p2.anti_forced(f, p1)
                p1.anti_forced(f, p2)

    @classmethod
    def momentum(cls) -> np.ndarray:
        """计算全局动量和"""
        if not cls.points:
            return np.zeros(3, dtype=Config.precision)
            
        # 预先分配数组并使用向量化操作
        velocities = np.array([p.v for p in cls.points], dtype=Config.precision)
        masses = np.array([p.m for p in cls.points], dtype=Config.precision).reshape(-1, 1)
        
        # 动量 = 质量 * 速度，求和
        return np.sum(velocities * masses, axis=0)

    @classmethod
    def run1(cls, t: float) -> None:
        """
        欧拉方法（一阶精度）
        
        :param t: 时间间隔
        """
        if not cls.points:
            return

        for p in cls.points:
            p.v += p.a * t
            p.pos += p.v * t
            p.old_a = p.a.copy()
            p.zero()
                
        cls.fps += 1

    @classmethod
    def run2(cls, t: float) -> None:
        """
        龙格-库塔法（二阶精度）

        :param t: 时间间隔
        """
        if not cls.points:
            return

        for p in cls.points:
            p.pos += p.v * t + 0.5 * p.a * t ** 2
            p.v += p.a * t
            p.old_a = p.a.copy()
            p.zero()

        cls.fps += 1

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
    def snapshot(cls, path: str = "state.pkl") -> None:
        """保存当前状态快照"""
        import pickle
        state = {
            "points": cls.points,
            "r_points": cls.r_points,
            "fps": cls.fps
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)

    @classmethod
    def load_snapshot(cls, path: str = "state.pkl") -> None:
        """加载状态快照"""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        cls.points = state["points"]
        cls.r_points = state["r_points"]
        cls.fps = state.get("fps", 0)


class DingPoint(Point):
    """定点，不参与力的计算"""
    def __init__(self, m: float, pos: Union[np.ndarray, tuple, list], v: Union[np.ndarray, tuple, list], 
                 r: float = None, color: Union[str, Tuple[int, int, int]] = "black"):
        # 调用父类初始化
        super().__init__(m, pos, v, r, color)
        # 重写加速度，使其始终为0
        self.a = np.zeros_like(self.v, dtype=Config.precision)
        self.old_a = self.a.copy()
        
    def forced(self, f: np.ndarray) -> None:
        """定点不受力"""
        pass
        
    def zero(self) -> None:
        """保持加速度为0"""
        pass