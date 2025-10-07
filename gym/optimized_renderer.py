import pygame
import numpy as np
from typing import Union, Tuple, List, Optional, Dict
from optimized_engine import Point, Config
import time

class Camera:
    def __init__(self, cam_pos: np.ndarray = None, look_pos: np.ndarray = None, k: float = 300):
        """
        创建一个相机
        :param cam_pos:相机位置
        :param look_pos: 注视点坐标
        :param k: 画面放大系数
        """
        if cam_pos is None:
            cam_pos = np.array([0, 0, -300], dtype=Config.precision)
        if look_pos is None:
            look_pos = np.array([0, 0, 0], dtype=Config.precision)
        self.cam = Point(1, cam_pos, [0, 0, 0])
        distance = max(np.linalg.norm(cam_pos - look_pos).astype(float), Config.r)
        self.look_pos = (look_pos - cam_pos) / distance
        self.k = k
        self.width = 800
        self.height = 600
        
    def set_look_pos(self, look_pos: np.ndarray) -> None:
        """
        设置相机视角
        :param look_pos: 注视点坐标
        """
        distance = max(np.linalg.norm(self.cam.pos - look_pos).astype(float), Config.r)
        self.look_pos = (look_pos - self.cam.pos) / distance
    
    def eye_space(self, pos: np.ndarray) -> np.ndarray:
        x = Point.eye(self.cam.pos, self.look_pos + self.cam.pos)
        d = Point.trans(pos, x, self.cam.pos)
        return d
    
    def dot_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        返回一个点被相机拍到后在屏幕上的位置
        :param pos: 点的坐标
        :return: 点在屏幕上的坐标，当无法进行透视变换时返回None
        """
        x = Point.eye(self.cam.pos, self.look_pos + self.cam.pos)
        d = Point.trans(pos, x, self.cam.pos)
        if d[2] >= 0:
            screen_pos = Point.perspective(d, [0, 0, 0], self.k)
            # 将坐标转换为屏幕坐标（原点在中心）
            return np.array([screen_pos[0] + self.width // 2, -screen_pos[1] + self.height // 2], dtype=int)
        return None

class Renderer:
    """使用Pygame的高性能渲染器"""
    def __init__(self, width: int = 800, height: int = 600, title: str = "Physics Simulation"):
        """
        初始化渲染器
        :param width: 窗口宽度
        :param height: 窗口高度
        :param title: 窗口标题
        """
        # 初始化Pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.camera = Camera()
        self.running = True
        self.show_fps = True
        self.show_acceleration = False
        self.show_velocity = False
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # 用于存储点和弹簧的渲染信息，减少重复计算
        self.point_render_data = []
        self.spring_render_data = []
    
    def handle_events(self) -> None:
        """处理用户输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_f:
                    self.show_fps = not self.show_fps
                elif event.key == pygame.K_a:
                    self.show_acceleration = not self.show_acceleration
                elif event.key == pygame.K_v:
                    self.show_velocity = not self.show_velocity
                # 相机控制
                elif event.key == pygame.K_w:
                    self.move_camera_forward(5)
                elif event.key == pygame.K_s:
                    self.move_camera_backward(5)
                elif event.key == pygame.K_a:
                    self.move_camera_left(5)
                elif event.key == pygame.K_d:
                    self.move_camera_right(5)
                elif event.key == pygame.K_SPACE:
                    self.move_camera_up(5)
                elif event.key == pygame.K_LCTRL:
                    self.move_camera_down(5)
                elif event.key == pygame.K_UP:
                    self.tilt_camera_up(0.05)
                elif event.key == pygame.K_DOWN:
                    self.tilt_camera_down(0.05)
                elif event.key == pygame.K_LEFT:
                    self.turn_camera_left(0.05)
                elif event.key == pygame.K_RIGHT:
                    self.turn_camera_right(0.05)
                elif event.key == pygame.K_EQUALS:
                    self.zoom_in(1.1)
                elif event.key == pygame.K_MINUS:
                    self.zoom_out(0.9)
    
    def move_camera_forward(self, distance: float) -> None:
        """向前移动相机"""
        # 只在x-z平面上移动
        direction = np.array([self.camera.look_pos[0], 0, self.camera.look_pos[2]], dtype=Config.precision)
        norm = np.linalg.norm(direction)
        if norm > Config.r:
            direction = direction / norm
        self.camera.cam.pos += direction * distance
    
    def move_camera_backward(self, distance: float) -> None:
        """向后移动相机"""
        self.move_camera_forward(-distance)
    
    def move_camera_left(self, distance: float) -> None:
        """向左移动相机"""
        # 计算向左的方向（垂直于注视方向）
        forward = np.array([self.camera.look_pos[0], 0, self.camera.look_pos[2]], dtype=Config.precision)
        left = np.cross([0, 1, 0], forward)
        norm = np.linalg.norm(left)
        if norm > Config.r:
            left = left / norm
        self.camera.cam.pos += left * distance
    
    def move_camera_right(self, distance: float) -> None:
        """向右移动相机"""
        self.move_camera_left(-distance)
    
    def move_camera_up(self, distance: float) -> None:
        """向上移动相机"""
        self.camera.cam.pos[1] += distance
    
    def move_camera_down(self, distance: float) -> None:
        """向下移动相机"""
        self.camera.cam.pos[1] -= distance
    
    def turn_camera_left(self, angle: float) -> None:
        """向左旋转相机"""
        # 在x-z平面上旋转注视方向
        x, z = self.camera.look_pos[0], self.camera.look_pos[2]
        new_x = x * np.cos(angle) - z * np.sin(angle)
        new_z = x * np.sin(angle) + z * np.cos(angle)
        self.camera.look_pos = np.array([new_x, self.camera.look_pos[1], new_z])
        # 归一化
        norm = np.linalg.norm(self.camera.look_pos)
        if norm > Config.r:
            self.camera.look_pos = self.camera.look_pos / norm
    
    def turn_camera_right(self, angle: float) -> None:
        """向右旋转相机"""
        self.turn_camera_left(-angle)
    
    def tilt_camera_up(self, angle: float) -> None:
        """向上倾斜相机"""
        # 计算在x-y平面上的旋转
        x, y, z = self.camera.look_pos
        # 先计算水平分量的长度
        horizontal_length = np.sqrt(x**2 + z**2)
        if horizontal_length > Config.r:
            # 计算当前的仰角
            current_angle = np.arcsin(y)
            # 限制仰角范围
            new_angle = min(np.pi/2 - Config.r, max(-np.pi/2 + Config.r, current_angle + angle))
            # 计算新的注视方向
            new_y = np.sin(new_angle)
            scale = np.cos(new_angle) / horizontal_length
            new_x = x * scale
            new_z = z * scale
            self.camera.look_pos = np.array([new_x, new_y, new_z])
    
    def tilt_camera_down(self, angle: float) -> None:
        """向下倾斜相机"""
        self.tilt_camera_up(-angle)
    
    def zoom_in(self, factor: float) -> None:
        """放大画面"""
        self.camera.k *= factor
    
    def zoom_out(self, factor: float) -> None:
        """缩小画面"""
        self.camera.k *= factor
    
    def calculate_render_data(self) -> None:
        """计算所有点和弹簧的渲染数据，避免重复计算"""
        self.point_render_data = []
        self.spring_render_data = []
        
        # 计算点的渲染数据
        for point in Point.points:
            screen_pos = self.camera.dot_pos(point.pos)
            if screen_pos is not None:
                # 根据距离调整点的大小
                eye_pos = self.camera.eye_space(point.pos)
                if eye_pos[2] > Config.r:
                    size = max(1, int(point.r * 2 * self.camera.k / eye_pos[2]))
                else:
                    size = max(1, int(point.r * 2))
                
                # 计算加速度和速度向量的终点位置
                acc_pos = None
                vel_pos = None
                
                if self.show_acceleration:
                    acc_end_pos = point.pos + point.old_a * 10  # 放大加速度向量
                    acc_screen_pos = self.camera.dot_pos(acc_end_pos)
                    if acc_screen_pos is not None:
                        acc_pos = acc_screen_pos
                
                if self.show_velocity:
                    vel_end_pos = point.pos + point.v * 5  # 放大速度向量
                    vel_screen_pos = self.camera.dot_pos(vel_end_pos)
                    if vel_screen_pos is not None:
                        vel_pos = vel_screen_pos
                
                self.point_render_data.append({
                    'pos': screen_pos,
                    'size': size,
                    'color': self._convert_color(point.color),
                    'acc_pos': acc_pos,
                    'vel_pos': vel_pos
                })
        
        # 计算弹簧的渲染数据
        for (p1, p2) in Point.r_points.keys():
            pos1 = self.camera.dot_pos(p1.pos)
            pos2 = self.camera.dot_pos(p2.pos)
            if pos1 is not None and pos2 is not None:
                self.spring_render_data.append({
                    'pos1': pos1,
                    'pos2': pos2,
                    'color': (0, 0, 0)  # 默认为黑色
                })
    
    def _convert_color(self, color: Union[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """将颜色转换为Pygame可用的格式"""
        if isinstance(color, str):
            # 简单的颜色名称映射
            color_map = {
                'black': (0, 0, 0),
                'white': (255, 255, 255),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'purple': (128, 0, 128),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255)
            }
            return color_map.get(color.lower(), (0, 0, 0))
        elif isinstance(color, tuple) and len(color) == 3:
            # 假设是RGB颜色
            return color
        else:
            return (0, 0, 0)  # 默认黑色
    
    def render(self) -> None:
        """渲染一帧画面"""
        # 清空屏幕
        self.screen.fill((255, 255, 255))  # 白色背景
        
        # 计算渲染数据
        self.calculate_render_data()
        
        # 渲染弹簧（先渲染弹簧，再渲染点，这样点会在弹簧上面）
        for spring in self.spring_render_data:
            pygame.draw.line(self.screen, spring['color'], spring['pos1'], spring['pos2'], 1)
        
        # 渲染点
        for point in self.point_render_data:
            pygame.draw.circle(self.screen, point['color'], point['pos'], point['size'])
            
            # 渲染加速度向量（红色）
            if point['acc_pos'] is not None:
                pygame.draw.line(self.screen, (255, 0, 0), point['pos'], point['acc_pos'], 2)
            
            # 渲染速度向量（蓝色）
            if point['vel_pos'] is not None:
                pygame.draw.line(self.screen, (0, 0, 255), point['pos'], point['vel_pos'], 2)
        
        # 显示FPS
        if self.show_fps:
            self._update_fps()
            fps_text = self.font.render(f"FPS: {self.fps}", True, (0, 0, 0))
            self.screen.blit(fps_text, (10, 10))
        
        # 更新显示
        pygame.display.flip()
    
    def _update_fps(self) -> None:
        """更新FPS计数"""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        if elapsed_time > 1.0:
            self.fps = int(self.frame_count / elapsed_time)
            self.frame_count = 0
            self.last_time = current_time
    
    def tick(self, fps: int = 60) -> None:
        """控制帧率"""
        self.clock.tick(fps)
    
    def is_running(self) -> bool:
        """检查渲染器是否仍在运行"""
        return self.running
    
    def close(self) -> None:
        """关闭渲染器"""
        pygame.quit()
    
    def play(self, fps: int = 60) -> None:
        """主渲染循环"""
        while self.running:
            self.handle_events()
            self.render()
            self.tick(fps)
        self.close()

# 为了兼容旧版本的API，提供一个与原Point.play类似的函数

def play(fps: int = 60, a: bool = False, v: bool = False, c: Point = None, 
         x: np.ndarray = None, a_zoom: float = 1, v_zoom: float = 1, k: float = 1) -> None:
    """
    兼容旧版本的渲染函数
    :param fps: 帧率
    :param a: 是否显示加速度标
    :param v: 是否显示速度标
    :param c: 参考系
    :param x: 线性变换矩阵
    :param a_zoom: 加速度标缩放系数
    :param v_zoom: 速度标缩放系数
    :param k: 透视变换放大系数
    """
    renderer = Renderer()
    renderer.show_acceleration = a
    renderer.show_velocity = v
    if c is not None:
        renderer.camera.cam = c
    if k is not None:
        renderer.camera.k = k
    renderer.play(fps)

# 更新Point类，使其使用新的渲染器

# 保存原始的ready方法
original_ready = Point.ready

# 重写ready方法以支持新的渲染器
Point.ready = lambda: None

# 提供一个使用新渲染器的play方法
Point.play = play