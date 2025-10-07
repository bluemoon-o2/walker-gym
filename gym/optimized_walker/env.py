import numpy as np
import time
from typing import List, Tuple, Optional, Union
from .core import Config, Point, to_data, DingPoint
from .renderer import Renderer, Scene, get_renderer


class Environment:
    """物理环境类，管理物理环境中的各种参数和物理规则"""
    def __init__(self, gravity: Union[np.ndarray, tuple, list] = (0, -9.8, 0),
                 damping: float = 0.99, ground: bool = True,
                 ground_level: float = -50, ground_restitution: float = 0.8,
                 air_resistance: float = 0.01, friction: float = 0.5,
                 time_step: float = 0.01, renderer: Optional[Renderer] = None):
        """
        初始化物理环境
        
        :param gravity: 重力向量
        :param damping: 阻尼系数，模拟能量损失
        :param ground: 是否启用地面
        :param ground_level: 地面高度
        :param ground_restitution: 地面弹性系数
        :param air_resistance: 空气阻力系数
        :param friction: 摩擦系数
        :param time_step: 物理模拟时间步长
        :param renderer: 渲染器实例
        """
        # 物理参数
        self.gravity = to_data(gravity)
        self.damping = damping
        self.ground = ground
        self.ground_level = ground_level
        self.ground_restitution = ground_restitution
        self.air_resistance = air_resistance
        self.friction = friction
        self.time_step = time_step
        
        # 环境中的物体
        self.points = []        # 物理点列表
        self.ding_points = []   # 定点列表
        self.springs = []       # 弹簧列表，存储为(点1, 点2, 原长, 劲度系数, 是否为绳)的元组
        
        # 渲染相关
        self.renderer = renderer or get_renderer()
        self.scene = Scene(self.renderer)
        
        # 运行状态
        self.running = False
        self.paused = False
        
        # 统计信息
        self.frame_count = 0
        self.start_time = 0
        self.last_time = 0
        
    def add_point(self, m: float, pos: Union[np.ndarray, tuple, list],
                 v: Union[np.ndarray, tuple, list] = (0, 0, 0), r: float = None,
                 color: Union[str, Tuple[int, int, int]] = "black") -> Point:
        """
        添加物理点到环境
        
        :param m: 质量
        :param pos: 位置
        :param v: 速度
        :param r: 半径
        :param color: 颜色
        :return: 创建的物理点对象
        """
        point = Point(m, pos, v, r, color)
        self.points.append(point)
        self.scene.add_point(point)
        return point
        
    def add_ding_point(self, m: float, pos: Union[np.ndarray, tuple, list],
                      v: Union[np.ndarray, tuple, list] = (0, 0, 0), r: float = None,
                      color: Union[str, Tuple[int, int, int]] = "red") -> DingPoint:
        """
        添加定点到环境
        
        :param m: 质量
        :param pos: 位置
        :param v: 速度
        :param r: 半径
        :param color: 颜色
        :return: 创建的定点对象
        """
        point = DingPoint(m, pos, v, r, color)
        self.ding_points.append(point)
        self.scene.add_point(point)
        return point
        
    def add_spring(self, point1: Point, point2: Point, x: float = None, k: float = 100,
                   string: bool = False) -> None:
        """
        添加弹簧到环境
        
        :param point1: 第一个点
        :param point2: 第二个点
        :param x: 弹簧原长，默认当前长度为原长
        :param k: 劲度系数
        :param string: 是否为绳型弹簧
        """
        if x is None:
            # 计算当前两点之间的距离作为原长
            x = np.linalg.norm(point1.pos - point2.pos).astype(Config.precision)
        
        # 存储弹簧信息
        self.springs.append((point1, point2, x, k, string))
        
        # 添加到场景的弹簧列表
        self.scene.add_spring(point1, point2)
        
    def batch_add_points(self, points_data: List[dict]) -> List[Point]:
        """
        批量添加物理点
        
        :param points_data: 点数据列表，每个元素是包含点参数的字典
        :return: 创建的物理点对象列表
        """
        points = []
        for data in points_data:
            point = self.add_point(**data)
            points.append(point)
        return points
        
    def batch_add_springs(self, springs_data: List[dict]) -> None:
        """
        批量添加弹簧
        
        :param springs_data: 弹簧数据列表，每个元素是包含弹簧参数的字典
        """
        for data in springs_data:
            self.add_spring(**data)
        
    def update_physics(self) -> None:
        """更新物理状态"""
        if not self.points:
            return
            
        # 清除所有点的加速度
        for point in self.points + self.ding_points:
            point.zero()

        # 应用重力
        for point in self.points:
            point.forced(self.gravity * point.m)
                
        # 应用弹簧力
        for point1, point2, x, k, string in self.springs:
            point1.resilience(point2, x, k, string)
                
        # 应用阻尼
        for point in self.points:
            point.v *= self.damping
        
        # 应用空气阻力
        for point in self.points:
            # 空气阻力与速度平方成正比，方向相反
            speed = np.linalg.norm(point.v).astype(Config.precision)
            drag_force = -0.5 * self.air_resistance * speed * point.v
            point.forced(drag_force)
        
        # 更新物理位置
        Point.run1(self.time_step)
        
        # 处理地面碰撞
        if self.ground:
            for point in self.points:
                # 检查是否与地面碰撞
                if point.pos[1] <= self.ground_level:
                    # 矫正位置
                    point.pos[1] = self.ground_level

                    # 应用弹性反弹
                    if point.v[1] < 0:
                        # 垂直速度反转并乘以弹性系数
                        point.v[1] = -point.v[1] * self.ground_restitution
                            
                        # 应用摩擦力减少水平速度
                        point.v[0] *= self.friction
                        point.v[2] *= self.friction
        
        # 更新统计信息
        self.frame_count += 1
        
    def update(self) -> None:
        """更新环境状态"""
        if self.running and not self.paused:
            self.update_physics()
            
        # 更新场景
        self.scene.update()
        
        # 检查是否继续运行
        if not self.renderer.running:
            self.running = False
            
    def run(self, steps: int = None, real_time: bool = True) -> None:
        """
        运行物理模拟
        
        :param steps: 运行的步数，如果为None则一直运行直到用户退出
        :param real_time: 是否按照真实时间运行
        """
        self.running = True
        self.start_time = time.time()
        self.last_time = self.start_time
        
        # 运行指定步数或直到用户退出
        step_count = 0
        while self.running and (steps is None or step_count < steps):
            # 更新环境
            self.update()
            
            # 渲染场景
            self.scene.renderer.render(self.points + self.ding_points, [(s[0], s[1]) for s in self.springs])
            
            # 如果不是实时运行，不限制帧率
            if not real_time:
                self.renderer.fps_limit = 0
            
            step_count += 1
        
        # 清理资源
        self.renderer.cleanup()
        
    def pause(self) -> None:
        """暂停模拟"""
        self.paused = True
        
    def resume(self) -> None:
        """恢复模拟"""
        self.paused = False
        
    def stop(self) -> None:
        """停止模拟"""
        self.running = False
        
    def get_statistics(self) -> dict:
        """
        获取模拟统计信息
        
        :return: 包含统计信息的字典
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            avg_fps = self.frame_count / elapsed_time
        else:
            avg_fps = 0
            
        return {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed_time,
            "avg_fps": avg_fps,
            "point_count": len(self.points) + len(self.ding_points),
            "spring_count": len(self.springs),
            "time_step": self.time_step
        }
        
    def save_state(self, path: str = "env_state.pkl") -> None:
        """保存环境状态"""
        import pickle
        state = {
            "points": self.points,
            "ding_points": self.ding_points,
            "springs": self.springs,
            "gravity": self.gravity,
            "damping": self.damping,
            "ground": self.ground,
            "ground_level": self.ground_level,
            "ground_restitution": self.ground_restitution,
            "air_resistance": self.air_resistance,
            "friction": self.friction,
            "time_step": self.time_step
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)
            
    def load_state(self, path: str = "env_state.pkl") -> None:
        """加载环境状态"""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        # 恢复环境状态
        self.points = state["points"]
        self.ding_points = state["ding_points"]
        self.springs = state["springs"]
        self.gravity = state["gravity"]
        self.damping = state["damping"]
        self.ground = state["ground"]
        self.ground_level = state["ground_level"]
        self.ground_restitution = state["ground_restitution"]
        self.air_resistance = state["air_resistance"]
        self.friction = state["friction"]
        self.time_step = state["time_step"]
        
        # 清空场景并重新添加所有对象
        self.scene = Scene(self.renderer)
        for point in self.points + self.ding_points:
            self.scene.add_point(point)
        for spring in self.springs:
            self.scene.add_spring(spring[0], spring[1])

class OptimizedEnvironment(Environment):
    """优化版本的物理环境，针对大规模质点模拟进行了优化"""
    def __init__(self, *args, **kwargs):
        """初始化优化环境"""
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 优化参数
        self.spatial_partition_size = 50  # 空间分区大小
        self.spatial_partitions = {}
        
        # 碰撞检测参数
        self.enable_spatial_partitioning = True  # 是否启用空间分区
        self.collision_margin = 1.0  # 碰撞检测的边距
        
        # 并行计算参数
        self.enable_parallel = True  # 是否启用并行计算
        
    def spatial_hash(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """
        计算空间哈希值
        
        :param pos: 位置坐标
        :return: 空间分区的键
        """
        x = int(pos[0] / self.spatial_partition_size)
        y = int(pos[1] / self.spatial_partition_size)
        z = int(pos[2] / self.spatial_partition_size)
        return (x, y, z)
        
    def build_spatial_partitions(self) -> None:
        """构建空间分区"""
        if not self.enable_spatial_partitioning:
            return
            
        # 清空现有分区
        self.spatial_partitions.clear()
        
        # 将所有点分配到空间分区
        for point in self.points:
            key = self.spatial_hash(point.pos)
            if key not in self.spatial_partitions:
                self.spatial_partitions[key] = []
            self.spatial_partitions[key].append(point)
            
    def get_nearby_points(self, point: Point, radius: float = None) -> List[Point]:
        """
        获取指定点附近的点
        
        :param point: 中心点
        :param radius: 搜索半径
        :return: 附近点的列表
        """
        if not self.enable_spatial_partitioning:
            # 如果没有启用空间分区，返回所有点
            return self.points.copy()
            
        if radius is None:
            radius = self.spatial_partition_size
            
        # 计算搜索的分区范围
        key = self.spatial_hash(point.pos)
        partitions = []
        
        # 搜索当前分区及其相邻分区
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    search_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if search_key in self.spatial_partitions:
                        partitions.extend(self.spatial_partitions[search_key])
                        
        return partitions
        
    def update_physics(self) -> None:
        """优化版本的物理更新，包含空间分区优化"""
        # 构建空间分区
        self.build_spatial_partitions()
        
        # 调用父类的物理更新方法
        super().update_physics()
        
    def detect_collisions(self) -> List[Tuple[Point, Point]]:
        """
        检测并返回所有碰撞的点对
        
        :return: 碰撞点对的列表
        """
        collisions = []
        
        if self.enable_spatial_partitioning:
            # 使用空间分区优化碰撞检测
            self.build_spatial_partitions()
            
            # 遍历每个分区
            for key, partition_points in self.spatial_partitions.items():
                # 检查分区内的点之间的碰撞
                for i in range(len(partition_points)):
                    for j in range(i + 1, len(partition_points)):
                        p1 = partition_points[i]
                        p2 = partition_points[j]
                        
                        # 检查是否发生碰撞
                        dist = np.linalg.norm(p1.pos - p2.pos).astype(Config.precision)
                        if dist < p1.r + p2.r + self.collision_margin:
                            collisions.append((p1, p2))
        else:
            # 简单的O(n²)碰撞检测
            for i in range(len(self.points)):
                for j in range(i + 1, len(self.points)):
                    p1 = self.points[i]
                    p2 = self.points[j]
                    
                    # 检查是否发生碰撞
                    dist = np.linalg.norm(p1.pos - p2.pos).astype(Config.precision)
                    if dist < p1.r + p2.r + self.collision_margin:
                        collisions.append((p1, p2))
                        
        return collisions

# 导出常用类和函数
export = {
    "Environment": Environment,
    "OptimizedEnvironment": OptimizedEnvironment,
    "Renderer": Renderer,
    "Scene": Scene,
    "Point": Point,
    "DingPoint": DingPoint,
    "Config": Config
}
