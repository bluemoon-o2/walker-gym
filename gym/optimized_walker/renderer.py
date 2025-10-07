import pygame
import numpy as np
from typing import Tuple, List, Optional, Union
from .core import Config, Point, DingPoint, to_data


class Camera:
    """相机控制类"""
    def __init__(self, pos: Union[np.ndarray, tuple, list] = (0, 0, 100),
                 rotate_speed: float = 0.05, move_speed: float = 5,
                 scale_speed: float = 1.2, sensitivity: float = 0.1):
        """
        初始化相机
        
        :param pos: 相机初始位置 [x, y, z]
        :param rotate_speed: 旋转速度
        :param move_speed: 移动速度
        :param scale_speed: 缩放速度
        :param sensitivity: 鼠标敏感度
        """
        self.pos = to_data(pos)
        self.rotate_speed = rotate_speed
        self.move_speed = move_speed
        self.scale_speed = scale_speed
        self.sensitivity = sensitivity
        
        # 相机朝向的旋转角度（欧拉角）
        self.theta = 0.0  # 绕y轴旋转
        self.phi = 0.0    # 绕x轴旋转
        
        # 投影参数
        self.fov = 60.0   # 视场角
        self.near = 0.1   # 近裁剪面
        self.far = 1000.0 # 远裁剪面
        
        # 鼠标状态跟踪
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        # 计算相机旋转矩阵
        self.update_rotation_matrix()

    def update_rotation_matrix(self) -> None:
        """更新相机旋转矩阵"""
        # 计算旋转矩阵（Z-Y-X欧拉角）
        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)
        sin_phi = np.sin(self.phi)
        cos_phi = np.cos(self.phi)
        
        # 绕x轴旋转（俯仰）
        rx = np.array([
            [1, 0, 0],
            [0, cos_phi, -sin_phi],
            [0, sin_phi, cos_phi]
        ], dtype=Config.precision)
        
        # 绕y轴旋转（偏航）
        ry = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ], dtype=Config.precision)
        
        # 复合旋转
        self.rotation_matrix = ry @ rx

    def rotate(self, delta_x: float, delta_y: float) -> None:
        """
        旋转相机视角
        
        :param delta_x: 鼠标水平移动量
        :param delta_y: 鼠标垂直移动量
        """
        self.theta += delta_x * self.rotate_speed * self.sensitivity
        self.phi += delta_y * self.rotate_speed * self.sensitivity
        
        # 限制俯仰角范围
        self.phi = max(-np.pi/2 + 0.1, min(np.pi/2 - 0.1, self.phi))
        
        # 更新旋转矩阵
        self.update_rotation_matrix()

    def move(self, direction: Union[np.ndarray, tuple, list]) -> None:
        """
        移动相机位置
        
        :param direction: 移动方向向量
        """
        direction = to_data(direction)
        
        # 应用旋转，使移动与相机朝向一致
        self.pos += self.rotation_matrix @ direction * self.move_speed

    def scale(self, factor: float) -> None:
        """
        缩放相机（沿着视线方向移动）
        
        :param factor: 缩放因子
        """
        # 沿着z轴（相机朝向）移动
        self.pos[2] *= factor
        
        # 确保不会穿过近裁剪面
        self.pos[2] = max(self.near * 2, self.pos[2])

class Renderer:
    """渲染器类，使用pygame进行高效渲染"""
    def __init__(self, width: int = 800, height: int = 600, title: str = "物理引擎模拟",
                 fps_limit: int = 60, background_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        初始化渲染器
        
        :param width: 窗口宽度
        :param height: 窗口高度
        :param title: 窗口标题
        :param fps_limit: 帧率限制
        :param background_color: 背景颜色
        """
        # 初始化pygame
        pygame.init()
        pygame.display.set_caption(title)
        
        # 设置窗口和渲染表面
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.fps_limit = fps_limit
        self.background_color = background_color
        
        # 创建相机对象
        self.camera = Camera()
        
        # 渲染选项
        self.show_springs = True
        self.show_vectors = False
        self.show_points = True
        self.show_fps = True
        
        # 字体设置
        self.font = pygame.font.SysFont("Arial", 16)
        
        # 批处理参数
        self.point_batch_size = 100
        self.spring_batch_size = 50
        
        # 程序运行状态
        self.running = True

    def project_point(self, point_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        将3D点投影到2D屏幕上
        
        :param point_pos: 3D点位置
        :return: 投影后的2D屏幕坐标，如果点在相机后方则返回None
        """
        # 应用相机旋转
        rel_pos = point_pos - self.camera.pos
        rotated = self.camera.rotation_matrix @ rel_pos
        
        # 透视投影
        if rotated[2] <= self.camera.near:
            return None  # 点在相机后方或近裁剪面内，不渲染
        
        # 投影公式：x' = x * scale / z, y' = y * scale / z
        scale = self.width * 0.5 / np.tan(np.radians(self.camera.fov) * 0.5)
        projected = rotated[:2] * scale / rotated[2]
        
        # 转换到屏幕坐标（原点在左上角）
        screen_x = self.width * 0.5 + projected[0]
        screen_y = self.height * 0.5 - projected[1]  # Y轴反转，因为pygame的Y轴向下
        
        # 检查是否在屏幕范围内
        if (screen_x < -100 or screen_x > self.width + 100 or
            screen_y < -100 or screen_y > self.height + 100):
            return None
        
        return np.array([screen_x, screen_y], dtype=np.int32)

    def draw_point(self, point: Point, size: Optional[float] = None) -> None:
        """
        绘制单个物理点
        
        :param point: 要绘制的物理点
        :param size: 点的大小，默认使用物理点的半径
        """
        # 投影到屏幕坐标
        screen_pos = self.project_point(point.pos)
        if screen_pos is None:
            return
            
        # 确定点的大小
        if size is None:
            # 考虑透视效果调整大小
            distance = np.linalg.norm(point.pos - self.camera.pos).astype(Config.precision)
            # 距离越远，点越小
            size = max(1, int(point.r * 2 * self.width / (distance * 0.1)))
            
        # 处理颜色
        if isinstance(point.color, str):
            # 字符串颜色名转换为RGB
            color = pygame.Color(point.color)
        else:
            # 假设是RGB元组
            color = point.color
        
        # 绘制点
        pygame.draw.circle(self.screen, color, screen_pos, size)

    def draw_spring(self, point1: Point, point2: Point, color: Tuple[int, int, int] = (100, 100, 100),
                   width: int = 1) -> None:
        """
        绘制连接两个点的弹簧/线
        
        :param point1: 第一个点
        :param point2: 第二个点
        :param color: 线的颜色
        :param width: 线的宽度
        """
        # 投影两个点到屏幕坐标
        p1_screen = self.project_point(point1.pos)
        p2_screen = self.project_point(point2.pos)
        
        # 如果两个点都可见，才绘制连接线
        if p1_screen is not None and p2_screen is not None:
            pygame.draw.line(self.screen, color, p1_screen, p2_screen, width)

    def draw_vector(self, point: Point, vector: np.ndarray, length_scale: float = 10.0,
                   color: Tuple[int, int, int] = (255, 0, 0), width: int = 2) -> None:
        """
        绘制向量（如速度、加速度向量）
        
        :param point: 向量起点
        :param vector: 向量数据
        :param length_scale: 向量长度缩放因子
        :param color: 向量颜色
        :param width: 向量线宽
        """
        # 投影起点到屏幕坐标
        start_pos = self.project_point(point.pos)
        if start_pos is None:
            return
            
        # 计算向量终点
        end_pos_3d = point.pos + vector * length_scale
        end_pos = self.project_point(end_pos_3d)
        
        if end_pos is not None:
            # 绘制向量线
            pygame.draw.line(self.screen, color, start_pos, end_pos, width)
            
            # 绘制箭头头部（简单的三角形）
            # 计算箭头方向
            direction = np.array(end_pos) - np.array(start_pos)
            direction_len = np.linalg.norm(direction)
            if direction_len > 0:
                direction = direction / direction_len
                
                # 计算箭头的两个侧边点
                arrow_size = 10
                side1 = np.array([-direction[1], direction[0]]) * arrow_size  # 垂直方向
                side2 = -side1
                
                # 计算箭头的三个点
                arrow_points = [
                    end_pos,
                    tuple(end_pos - direction * arrow_size + side1),
                    tuple(end_pos - direction * arrow_size + side2)
                ]
                
                # 绘制箭头头部
                pygame.draw.polygon(self.screen, color, arrow_points)

    def batch_draw_points(self, points: List[Point]) -> None:
        """
        批量绘制多个点，提高渲染效率
        
        :param points: 要点列表
        """
        # 预处理点数据
        for i in range(0, len(points), self.point_batch_size):
            batch = points[i:i+self.point_batch_size]
            for point in batch:
                self.draw_point(point)

    def batch_draw_springs(self, springs: List[Tuple[Point, Point]]) -> None:
        """
        批量绘制多个弹簧/线，提高渲染效率
        
        :param springs: 弹簧列表，每个元素是(点1, 点2)的元组
        """
        # 预处理弹簧数据
        for i in range(0, len(springs), self.spring_batch_size):
            batch = springs[i:i+self.spring_batch_size]
            for point1, point2 in batch:
                self.draw_spring(point1, point2)

    def draw_fps(self) -> None:
        """绘制当前帧率"""
        fps = self.clock.get_fps()
        fps_text = self.font.render(f"FPS: {fps:.1f}", True, (0, 0, 0))
        self.screen.blit(fps_text, (10, 10))

    def process_events(self) -> None:
        """处理用户输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                # 键盘控制
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_w:
                    # 向前移动
                    self.camera.move((0, 0, -1))
                elif event.key == pygame.K_s:
                    # 向后移动
                    self.camera.move((0, 0, 1))
                elif event.key == pygame.K_a:
                    # 向左移动
                    self.camera.move((-1, 0, 0))
                elif event.key == pygame.K_d:
                    # 向右移动
                    self.camera.move((1, 0, 0))
                elif event.key == pygame.K_SPACE:
                    # 向上移动
                    self.camera.move((0, 1, 0))
                elif event.key == pygame.K_LSHIFT:
                    # 向下移动
                    self.camera.move((0, -1, 0))
                elif event.key == pygame.K_p:
                    # 切换显示点
                    self.show_points = not self.show_points
                elif event.key == pygame.K_r:
                    # 切换显示弹簧
                    self.show_springs = not self.show_springs
                elif event.key == pygame.K_v:
                    # 切换显示向量
                    self.show_vectors = not self.show_vectors
                elif event.key == pygame.K_f:
                    # 切换显示帧率
                    self.show_fps = not self.show_fps
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 鼠标滚轮缩放
                if event.button == 4:  # 滚轮上滚
                    self.camera.scale(1/self.camera.scale_speed)
                elif event.button == 5:  # 滚轮下滚
                    self.camera.scale(self.camera.scale_speed)
                elif event.button == 1:  # 左键拖动旋转
                    self.camera.mouse_dragging = True
                    self.camera.last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                # 结束鼠标拖动
                if event.button == 1:
                    self.camera.mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                # 鼠标拖动旋转
                if self.camera.mouse_dragging:
                    current_pos = pygame.mouse.get_pos()
                    delta_x = current_pos[0] - self.camera.last_mouse_pos[0]
                    delta_y = current_pos[1] - self.camera.last_mouse_pos[1]
                    
                    # 旋转相机
                    self.camera.rotate(delta_x, delta_y)
                    
                    # 更新鼠标位置
                    self.camera.last_mouse_pos = current_pos

    def clear(self) -> None:
        """清空屏幕"""
        self.screen.fill(self.background_color)

    def flip(self) -> None:
        """更新显示"""
        pygame.display.flip()

    def tick(self) -> None:
        """控制帧率"""
        self.clock.tick(self.fps_limit)

    def render(self, points: List[Point], springs: List[Tuple[Point, Point]] = None) -> None:
        """
        渲染一帧画面
        
        :param points: 要渲染的点列表
        :param springs: 要渲染的弹簧列表
        """
        # 清空屏幕
        self.clear()
        
        # 处理输入事件
        self.process_events()
        
        # 绘制弹簧
        if springs is not None and self.show_springs:
            self.batch_draw_springs(springs)

        # 绘制点和向量
        if self.show_points:
            self.batch_draw_points(points)

        # 绘制向量
        if self.show_vectors:
            for point in points:
                # 绘制速度向量（蓝色）
                self.draw_vector(point, point.v, 1.0, (0, 0, 255))
                # 绘制加速度向量（红色）
                self.draw_vector(point, point.a, 10.0, (255, 0, 0))
        
        # 绘制帧率
        if self.show_fps:
            self.draw_fps()
        
        # 更新显示
        self.flip()
        
        # 控制帧率
        self.tick()

    def cleanup(self) -> None:
        """清理资源"""
        pygame.quit()

class Scene:
    """场景管理类"""
    def __init__(self, renderer: Optional[Renderer] = None):
        """
        初始化场景
        
        :param renderer: 渲染器实例，如果为None则创建默认渲染器
        """
        if renderer is None:
            self.renderer = Renderer()
        else:
            self.renderer = renderer
        
        # 场景中的点和弹簧
        self.points = []
        self.springs = []
        
        # 物理参数
        self.time_step = 0.01
        self.gravity = np.array([0, -9.8, 0], dtype=Config.precision)
        self.damping = 0.99  # 阻尼系数，模拟空气阻力
        
        # 运行状态
        self.running = True

    def add_point(self, point: Point) -> None:
        """添加点到场景"""
        self.points.append(point)

    def add_spring(self, point1: Point, point2: Point) -> None:
        """添加弹簧到场景"""
        self.springs.append((point1, point2))

    def apply_gravity(self) -> None:
        """应用重力"""
        for point in self.points:
            # 跳过定点
            if isinstance(point, Point) and not isinstance(point, DingPoint):
                point.forced(self.gravity * point.m)

    def apply_damping(self) -> None:
        """应用阻尼"""
        for point in self.points:
            # 阻尼力与速度成正比，方向相反
            point.forced(-self.damping * point.v)

    def update_physics(self) -> None:
        """更新物理状态"""
        # 应用重力
        self.apply_gravity()
        
        # 应用阻尼
        self.apply_damping()
        
        # 处理弹簧力
        for point1, point2 in self.springs:
            point1.resilience(point2)
        
        # 更新物理位置
        Point.run1(self.time_step)

    def update(self) -> None:
        """更新场景"""
        # 更新物理状态
        self.update_physics()
        
        # 检查渲染器是否仍在运行
        if not self.renderer.running:
            self.running = False

    def play(self) -> None:
        """运行场景"""
        while self.running:
            # 更新场景
            self.update()
            
            # 渲染场景
            self.renderer.render(self.points, self.springs)
        
        # 清理资源
        self.renderer.cleanup()

    def ready(self) -> None:
        """准备场景（旧接口兼容性）"""
        pass

    def view(self) -> None:
        """查看场景（旧接口兼容性）"""
        pass

    def keymove(self) -> None:
        """键盘控制（旧接口兼容性）"""
        pass

# 为了兼容旧的代码接口，提供一个全局的渲染器实例
_renderer_instance = None

def get_renderer() -> Renderer:
    """获取全局渲染器实例"""
    global _renderer_instance
    if _renderer_instance is None:
        _renderer_instance = Renderer()
    return _renderer_instance


def set_renderer(renderer: Renderer) -> None:
    """设置全局渲染器实例"""
    global _renderer_instance
    _renderer_instance = renderer