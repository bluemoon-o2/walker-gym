"""优化版牛顿力学物理环境模拟包"""

# 导出核心模块
def _export_module(module_name):
    """动态导入并导出模块的内容"""
    import importlib
    module = importlib.import_module(f".{module_name}", package=__name__)
    if hasattr(module, "export"):
        globals().update(module.export)

# 导出核心模块的公共接口
_export_module("core")
_export_module("renderer")
_export_module("env")
_export_module("walker")

# 直接导入常用类和函数，以便用户可以直接从包中导入
try:
    from .core import Config, Point, DingPoint, to_data
    from .renderer import Renderer, Scene, get_renderer, set_renderer
    from .env import Environment, OptimizedEnvironment
    from .walker import Muscle, Skeleton, Creature, Brain
    from .walker import test, leg2, box, balance1, balance2, balance3, humanb, insect
    
    # 定义包的公共接口
    __all__ = [
        # 核心类
        "Config", "Point", "DingPoint", "to_data",
        # 渲染器类
        "Renderer", "Scene", "get_renderer", "set_renderer",
        # 环境类
        "Environment", "OptimizedEnvironment",
        # 生物类
        "Muscle", "Skeleton", "Creature", "Brain",
        # 生物结构函数
        "test", "leg2", "box", "balance1", "balance2", "balance3", "humanb", "insect"
    ]
except ImportError as e:
    # 在导入失败时提供更友好的错误信息
    print(f"Warning: Failed to import some components: {e}")

# 包版本信息
__version__ = "1.0.0"
__author__ = "Optimized Physics Engine Team"
__license__ = "MIT"

# 优化版本说明
__description__ = """
优化版牛顿力学物理环境模拟包，具有以下改进：

1. 渲染层优化：
   - 使用pygame替代turtle进行高效渲染
   - 实现了相机控制、3D透视投影
   - 支持批量绘制以提高性能

2. 计算密集型模块优化：
   - 使用numba对核心计算函数进行JIT编译加速
   - 实现了并行计算以充分利用多核CPU
   - 优化了向量运算，使用numpy的向量化操作

3. 数据结构和算法优化：
   - 实现了空间分区以加速碰撞检测
   - 使用哈希表存储弹簧连接关系
   - 批量处理物理计算以减少循环开销

4. 资源占用优化：
   - 使用float32替代float64以减少内存占用
   - 实现了帧率限制和按需渲染
   - 优化了内存管理和对象引用
"""

# 默认配置
_DEFAULT_CONFIG = {
    "use_numba": True,
    "precision": "float32",
    "batch_size": 100,
    "enable_spatial_partitioning": True
}

def configure(**kwargs):
    """
    配置优化版物理引擎的参数
    
    :param use_numba: 是否使用numba加速
    :param precision: 浮点数精度（"float32"或"float64"）
    :param batch_size: 批处理大小
    :param enable_spatial_partitioning: 是否启用空间分区
    """
    from .core import Config
    from numpy import float32, float64
    
    # 更新配置参数
    if "use_numba" in kwargs:
        Config.use_numba = kwargs["use_numba"]
    
    if "precision" in kwargs:
        if kwargs["precision"] == "float64":
            Config.precision = float64
        else:
            Config.precision = float32
    
    if "batch_size" in kwargs:
        Config.batch_size = kwargs["batch_size"]
    
    # 更新环境配置
    if "enable_spatial_partitioning" in kwargs:
        try:
            from .env import OptimizedEnvironment
            # 可以在这里设置全局的默认环境配置
        except ImportError:
            pass
    
    return Config