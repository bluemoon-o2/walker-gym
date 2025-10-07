import pickle
import turtle
import numpy as np
from typing import Union, Tuple, List


class Config:
    precision = np.float32
    r = 16e-36
    e = 16e-20
    k = 8.99e9
    g = 9.8

def to_data(data: Union[np.ndarray, tuple, list]) -> np.ndarray:
    """规整数据"""
    if isinstance(data, (tuple, list)):
        return np.array(data, dtype=Config.precision)
    if isinstance(data, np.ndarray):
        return data.astype(Config.precision)
    else:
        raise TypeError(f"Data must be a numpy array, tuple, list (not {type(data).__name__})")


class Point:
    """点对象"""

    points = []   # 记录所有点
    r_points = {} # 储存需要连成弹簧的点
    fps = 0       # 帧数记录

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
        self.pos = to_data(pos)
        self.v = to_data(v)
        self.a = np.zeros_like(self.v, dtype=Config.precision)
        if r is None:
            r = m ** 0.3
        self.r = r
        self.old_a = self.a
        self.color = color
        self.e = e
        Point.points.append(self)

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
        # 等价于sqrt(x²+y²+z²)
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
    def gravity(cls) -> None:
        """全局引力"""
        for i in range(len(Point.points)):
            for j in range(i + 1, len(Point.points)):
                r = np.linalg.norm(Point.points[i].pos - Point.points[j].pos).astype(float)
                r = max(r, Config.r)
                f = -Config.g * Point.points[i].m * Point.points[j].m / (r ** 2)
                Point.points[j].anti_forced(f, Point.points[i])
                Point.points[i].anti_forced(f, Point.points[j])

    @classmethod
    def coulomb(cls) -> None:
        """全局静电力"""
        for i in range(len(Point.points)):
            for j in range(i + 1, len(Point.points)):
                r = np.linalg.norm(Point.points[i].pos - Point.points[j].pos).astype(float)
                r = max(r, Config.r)
                f = -Config.k * Point.points[i].e * Point.points[j].e / (r ** 2)
                Point.points[j].anti_forced(f, Point.points[i])
                Point.points[i].anti_forced(f, Point.points[j])

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
        m_sum = np.zeros_like(Point.points[0].v, dtype=Config.precision)
        for i in Point.points:
            m_sum += i.v * i.m
        return m_sum

    @classmethod
    def run1(cls, t: float) -> None:
        """
        欧拉方法（一阶精度）
        :param t: 时间间隔
        """
        for p in Point.points:
            p.v += p.a * t
            p.pos += p.v * t
            p.old_a = p.a[:]
            p.zero()

    @classmethod
    def run2(cls, t: float) -> None:
        """
        二阶龙格 - 库塔法（二阶精度）
        :param t: 时间间隔
        """
        for p in Point.points:
            p.pos += p.v * t + 0.5 * p.a * t ** 2
            p.v += p.a * t
            p.old_a = p.a[:]
            p.zero()

    @classmethod
    def ready(cls) -> None:
        """初始化显示模块"""
        turtle.tracer(0)
        turtle.penup()
        turtle.hideturtle()

    @classmethod
    def snapshot(cls, path="state.pkl") -> None:
        """保存快照"""
        state = {"points": cls.points, "r_points": cls.r_points}
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)

    @classmethod
    def backup(cls, path="state.pkl") -> None:
        """读取快照"""
        with open(path, "rb") as f:
            state = pickle.load(f)
        cls.points = state["points"]
        cls.r_points = state["r_points"]

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

    @classmethod
    def play(cls, fps: int = 1, a: bool = False, v: bool = False, c: 'Point' = None,
             x: np.ndarray = None, a_zoom: float = 1, v_zoom: float = 1, k: float = 1) -> None:
        """
        渲染当前状态
        :param fps: 跳过的帧数
        :param a: 是否显示加速度标
        :param v: 是否显示速度标
        :param c: 参考系
        :param x: 线性变换矩阵
        :param a_zoom: 加速度标缩放系数
        :param v_zoom: 速度标缩放系数
        :param k: 透视变换放大系数
        """
        if c is None:
            c = Point(0, [0, 0, 0], [0, 0, 0], 0)
        project = lambda y : cls.perspective(y, c.pos, k)
        if x is None:
            x = np.eye(3, dtype=Config.precision)
        if cls.fps % fps == 0:
            # 弹簧绘制
            for i in cls.r_points:
                turtle.color("black")
                dr0 = cls.trans(i[0].pos, x, c.pos)
                dr1 = cls.trans(i[1].pos, x, c.pos)
                if dr0[2] <= 0 or dr1[2] <= 0:
                    continue
                dr0 = project(dr0)
                dr1 = project(dr1)
                turtle.goto(dr0[0], dr0[1])
                turtle.pendown()
                turtle.goto(dr1[0], dr1[1])
                turtle.penup()
            Point.r_points = []
            # 点绘制
            for i in cls.points:
                d = cls.trans(i.pos, x, c.pos)
                if d[2] <= 0:
                    continue
                d2 = project(d)
                turtle.goto(d2[0], d2[1])
                turtle.dot(i.r * 2 / d[2] * k if d[2] != 0 else i.r * 2, i.color)
                if a:
                    da = x @ (i.pos + (i.old_a - c.old_a) * a_zoom)
                    if da[2] <= 0:
                        continue
                    da = project(da)
                    turtle.pencolor("red")
                    turtle.goto(d2[0], d2[1])
                    turtle.pendown()
                    turtle.goto(da[0], da[1])
                    turtle.penup()
                    turtle.pencolor("black")
                if v:
                    dv = x @ (i.pos + (i.v - c.v) * v_zoom)
                    if dv[2] <= 0:
                        continue
                    dv = project(dv)
                    turtle.pencolor("blue")
                    turtle.goto(d2[0], d2[1])
                    turtle.pendown()
                    turtle.goto(dv[0], dv[1])
                    turtle.penup()
                    turtle.pencolor("black")

            turtle.update()
            turtle.clear()
        Point.fps += 1


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
        d= Phy.trans(pos, self.cam.p, x)
        if d[2] >= 0:
            return Phy.perspective(d,[0,0,0],self.k)
        return None

        @classmethod
        def ready(cls):
            Point.ready()

        def tplay(self,a=False,v=False,azoom=1,vzoom=1,zuobiaoxian=False):
            """
            使用turtle的相机显示模块（只显示1帧，需循环调用）
            :param a: bool 是否显示加速度标
            :param v: bool 是否显示速度标
            :param azoom: float 加速度标放大系数
            :param vzoom: float 速度标放大系数
            :param zuobiaoxian: bool 是否显示迷你坐标线
            :return: None
            """
            x = Phy.eye(self.cam.p, [self.relalookpos[0] + self.cam.p[0],
                                     self.relalookpos[1] + self.cam.p[1],
                                     self.relalookpos[2] + self.cam.p[2]])
            if zuobiaoxian:
                xian = [Phy.translinear([100, 0, 0], x),
                        Phy.translinear([0, 100, 0], x),
                        Phy.translinear([0, 0, 100], x)]
                turtle.goto(xian[2][0], xian[2][1])
                turtle.dot(3, "red")
                for i in range(len(xian)):
                    turtle.pencolor("black")
                    turtle.goto(0, 0)
                    turtle.pd()
                    turtle.goto(xian[i][0], xian[i][1])
                    turtle.pu()
            Phy.play(a=a, v=v, c=self.cam, x=x, a_zoom=azoom, v_zoom=vzoom, k=self.k)

        def movecam(self, stepsize=1, camstepsize=0.02):
            '''
            通过turtle键盘控制移动相机与转换视角
            前进：w
            后退：s
            左移：a
            右移：d
            上移：空格
            下移：左Control
            左转：左箭头
            右转：右箭头
            上仰：上箭头
            下俯：下箭头
            放大：]
            缩小：[
            :param stepsize: float 相机移动步长
            :param camstepsize: float 相机视角转动步长
            :return: None
            '''
            def fw():
                dl = (self.relalookpos[0] ** 2 + self.relalookpos[2] ** 2) ** 0.5
                self.cam.p[0] += self.relalookpos[0] / dl * stepsize
                self.cam.p[2] += self.relalookpos[2] / dl * stepsize
            def bw():
                dl = (self.relalookpos[0] ** 2 + self.relalookpos[2] ** 2) ** 0.5
                self.cam.p[0] -= self.relalookpos[0] / dl * stepsize
                self.cam.p[2] -= self.relalookpos[2] / dl * stepsize
            def le():
                dl = (self.relalookpos[0] ** 2 + self.relalookpos[2] ** 2) ** 0.5
                self.cam.p[0] -= self.relalookpos[2] / dl * stepsize
                self.cam.p[2] -= -self.relalookpos[0] / dl * stepsize
            def ri():
                dl = (self.relalookpos[0] ** 2 + self.relalookpos[2] ** 2) ** 0.5
                self.cam.p[0] += self.relalookpos[2] / dl * stepsize
                self.cam.p[2] += -self.relalookpos[0] / dl * stepsize
            def zp():
                self.cam.p[1] += stepsize
            def zn():
                self.cam.p[1] -= stepsize
            def cu():
                self.relalookpos[1] += camstepsize
            def cd():
                self.relalookpos[1] -= camstepsize
            def cl():
                dl = (self.relalookpos[0] ** 2 + self.relalookpos[2] ** 2) ** 0.5
                self.relalookpos[0] -= self.relalookpos[2] / dl * camstepsize
                self.relalookpos[2] -= -self.relalookpos[0] / dl * camstepsize
            def cr():
                dl = (self.relalookpos[0] ** 2 + self.relalookpos[2] ** 2) ** 0.5
                self.relalookpos[0] += self.relalookpos[2] / dl * camstepsize
                self.relalookpos[2] += -self.relalookpos[0] / dl * camstepsize
            def zp2():
                self.relalookpos[2] += camstepsize
            def zn2():
                self.relalookpos[2] -= camstepsize
            def zin():
                self.k*=1.1
            def zout():
                self.k*=0.9

            turtle.onkeypress(fw, key="w")
            turtle.onkeypress(bw, key="s")
            turtle.onkeypress(le, key="a")
            turtle.onkeypress(ri, key="d")
            turtle.onkeypress(zp, key="space")
            turtle.onkeypress(zn, key="Control_L")
            turtle.onkeypress(cu, key="Up")
            turtle.onkeypress(cd, key="Down")
            turtle.onkeypress(cl, key="Left")
            turtle.onkeypress(cr, key="Right")
            turtle.onkeypress(zp2, key="u")
            turtle.onkeypress(zn2, key="o")
            turtle.onkeypress(zin, key="]")
            turtle.onkeypress(zout, key="[")
            turtle.listen()

    class tgraph:
        '''
        使用turtle实现的图表显示，使用前先创建对象
        在循环里使用draw
        '''
        def __init__(self):
            self.biao=[]
            self.zhenshu=0

        def clean(self):
            '''
            清空图表
            :return: None
            '''
            self.__init__()

        def draw(self,inx,iny,dis,chang=200,kx=1,ky=1,tiao=1,color="black",phyon=True,bi=False):
            """
            使用turtle实现的图表显示
            :param inx: float 点的x坐标，若希望图表不会移动，此处为None
            :param iny: float 点的y坐标
            :param dis: list[x,y] 坐标原点位置
            :param chang: float 图表长度
            :param kx: float x放大系数
            :param ky: float y放大系数
            :param tiao: float 每隔多少次采样
            :param color: list(r,g,b) 颜色
            :param phyon: bool 是否使用Phy.tplay
            :param bi: bool 是否在点之间画线
            :return: None
            """
            import turtle
            if phyon is False:
                Point.ready()

            if self.zhenshu%tiao==0:
                if inx is None:
                    self.biao.append([len(self.biao), iny])
                else:
                    self.biao.append([inx, iny])
            while len(self.biao) > chang:
                self.biao.pop(0)

            if inx is None:
                if bi is True:
                    turtle.pencolor(color)
                    turtle.goto(dis[0], dis[1] + self.biao[0][1] * ky)
                    turtle.pendown()
                for i in range(len(self.biao)):
                    turtle.goto(dis[0] + i * kx, dis[1] + self.biao[i][1] * ky)
                    turtle.dot(2, color)
                if bi is True:
                    turtle.penup()
            else:
                if bi is True:
                    turtle.pencolor(color)
                    turtle.goto(dis[0] + self.biao[0][0] * kx, dis[1] + self.biao[0][1] * ky)
                    turtle.pendown()
                for i in range(len(self.biao)):
                    turtle.goto(dis[0] + self.biao[i][0] * kx, dis[1] + self.biao[i][1] * ky)
                    turtle.dot(2, color)
                if bi is True:
                    turtle.penup()

            if phyon is False:
                turtle.update()
                turtle.clear()
            self.zhenshu+=1


class DingPoint(Point): #定点，不参与力的计算
    def __init__(self, m, v, p, r=None, color="black"):
        self.m = m
        self.v = v
        self.p = p
        self.a = [0, 0, 0]
        if r is None:
            r = m ** 0.3
        self.r = r
        self.axianshi = [0,0,0]
        self.color = color

class Scene:
    """
    场景，运行object
    """

    objs = [] #装着所有object的表
    camara = [0, 0, -1] #相机位置
    k = 1 #镜头放大参数

    @classmethod
    def ready(cls) -> None:
        turtle.tracer(0)
        turtle.penup()
        turtle.hideturtle()

    @classmethod
    def update(cls) -> None:
        Scene.objs.sort(key=lambda x: x.p[2], reverse=True)

    @classmethod
    def view(cls, p, camara, k) -> tuple:
        """
        小孔成像变换
        :param p: list[x,y,z]被拍摄点位置
        :param camara: list[x,y,z]摄相机位置
        :param k: float镜头放大参数（k>0）
        :return: tuple(dx, dy)变换后坐标
        """
        viewlength = camara[2] - p[2]
        if viewlength==0:
            viewlength=0.0000001
        dx = (camara[0] - p[0]) / viewlength * k
        dy = (camara[1] - p[1]) / viewlength * k
        return dx, dy

    @classmethod
    def play(cls,t):
        """
        使用turtle的显示模块（只显示1帧，需和run一起循环调用）
        :param t: 运行1帧中的时间
        :return: None
        """
        import turtle
        for i in Scene.objs:
            if i.p[2] <=Scene.camara[2]:
                continue
            i.draw()
        turtle.update()
        turtle.clear()
        Phy.run1(t)

    @classmethod
    def keymove(cls):
        import turtle
        def zf():
            Scene.k *= 1.1

        def zb():
            Scene.k *= 0.9

        def f():
            Scene.camara[2] += 1

        def b():
            Scene.camara[2] -= 1

        def l():
            Scene.camara[0] -= 100

        def r():
            Scene.camara[0] += 100

        def u():
            Scene.camara[1] += 100

        def d():
            Scene.camara[1] -= 100

        def reset(x, y):
            Scene.k = 1
            Scene.camara = [0, 0, -1]

        turtle.onkeypress(zf, key="=")
        turtle.onkeypress(zb, key="-")
        turtle.onkeypress(f, key="w")
        turtle.onkeypress(b, key="s")
        turtle.onkeypress(l, key="Left")
        turtle.onkeypress(r, key="Right")
        turtle.onkeypress(u, key="Up")
        turtle.onkeypress(d, key="Down")
        turtle.onscreenclick(reset)
        turtle.listen()


class Object:
    """
    对Phy的封装
    """
    def __init__(self, color=(0, 0, 0)):
        self.p = None
        self.biao = []
        self.color = color
        Scene.objs.append(self)

    def tri(self, d, h, p, v=None, m=1, color="black"):
        """
        三角形对象
        :param d: 底边长
        :param h: 高长
        :param p: 位置（左下角）
        :param v: 速度
        :param m: 质量
        :param color: 颜色
        :return: None
        """
        if v is None:
            v = [0, 0, 0]
        self.biao = [Phy(m, v, [p[0], p[1], p[2]]),
                     Phy(m, v, [p[0] + d, p[1], p[2]]),
                     Phy(m, v, [p[0] + d/2, p[1] + h, p[2]]),
                     Phy(m, v, p),
                     ]
        self.color = color
        self.p=p

    def fang(self, r, p, v=None, m=1, color="black"):
        '''
        自己变为正方形对象
        :param r: 边长
        :param p: 位置（左下角）
        :param v: 速度
        :param m: 质量
        :param color: 颜色
        :return: None
        '''
        if v is None:
            v = [0, 0, 0]
        self.biao = [Phy(m, v, [p[0], p[1], p[2]]),
                     Phy(m, v, [p[0] + r, p[1], p[2]]),
                     Phy(m, v, [p[0] + r, p[1] + r, p[2]]),
                     Phy(m, v, [p[0], p[1] + r, p[2]]),
                     Phy(m, v, p),
                     ]
        self.color = color
        self.p=p

    def cfang(self, c,f, p, v=None, m=1, color="black"):
        '''
        自己变为正方形对象
        :param c: 长
        :param f: 宽
        :param p: 位置（左下角）
        :param v: 速度
        :param m: 质量
        :param color: 颜色
        :return: None
        '''
        if v is None:
            v = [0, 0, 0]
        self.biao = [Phy(m, v, [p[0], p[1], p[2]]),
                     Phy(m, v, [p[0] + c, p[1], p[2]]),
                     Phy(m, v, [p[0] + c, p[1] + f, p[2]]),
                     Phy(m, v, [p[0], p[1] + f, p[2]]),
                     Phy(m, v, p),
                     ]
        self.color = color
        self.p=p

    def draw(self) -> None:
        turtle.fillcolor(self.color)
        turtle.begin_fill()
        for i in self.biao:
            turtle.goto(Scene.view(i.p, Scene.camara, Scene.k))
        turtle.end_fill()


