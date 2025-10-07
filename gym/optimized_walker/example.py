#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""优化版物理引擎示例"""

import sys
import time
import numpy as np
from .core import Config
from .env import Environment, OptimizedEnvironment
from .walker import Muscle, Skeleton, Creature, Brain
from .walker import test, leg2, box, balance1, balance2, balance3, humanb, insect


# 提供一个简单的示例运行函数
def run_example(example_type: str = "leg2", **kwargs):
    """
    运行一个示例模拟
    
    :param example_type: 示例类型（"test", "leg2", "box", "balance1", "balance2", "balance3", "humanb", "insect"）
    :param kwargs: 传递给环境和渲染器的参数
    """
    # 创建环境
    env = Environment(**kwargs)
    
    # 根据示例类型创建生物
    creature_functions = {
        "test": test,
        "leg2": leg2,
        "box": box,
        "balance1": balance1,
        "balance2": balance2,
        "balance3": balance3,
        "humanb": humanb,
        "insect": insect
    }
    
    if example_type in creature_functions:
        creature = creature_functions[example_type](env)
        print(f"Created {example_type} creature")
    else:
        # 如果示例类型不存在，默认使用双足生物
        creature = leg2(env)
        print(f"Unknown example type '{example_type}', using 'leg2' instead")
    
    # 运行模拟
    print("Starting simulation...")
    print("Controls:")
    print("  WASD + Space/Shift: Move camera")
    print("  Mouse drag: Rotate camera")
    print("  Mouse wheel: Zoom in/out")
    print("  P: Toggle points display")
    print("  R: Toggle springs display")
    print("  V: Toggle vectors display")
    print("  F: Toggle FPS display")
    print("  Escape: Exit")
    
    env.run()
    
    # 模拟结束后显示统计信息
    stats = env.get_statistics()
    print("\nSimulation statistics:")
    print(f"  Frames: {stats['frame_count']}")
    print(f"  Elapsed time: {stats['elapsed_time']:.2f} seconds")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")
    print(f"  Points count: {stats['point_count']}")
    print(f"  Springs count: {stats['spring_count']}")
    
    # 如果有生物，计算并显示适应度
    if 'creature' in locals():
        fitness = creature.evaluate_fitness()
        print(f"  Creature fitness: {fitness:.2f}")


# 运行性能测试的函数
def run_performance_test(num_points: int = 100, steps: int = 1000, use_optimizations: bool = True):
    """
    运行性能测试
    
    :param num_points: 点的数量
    :param steps: 运行的步数
    :param use_optimizations: 是否使用优化
    :return: 测试结果字典
    """
    # 保存当前配置
    original_config = {
        "precision": Config.precision,
        "batch_size": Config.batch_size
    }
    
    # 设置测试配置
    if use_optimizations:
        from numpy import float32
        Config.precision = float32
        Config.batch_size = 100
    else:
        from numpy import float64
        Config.precision = float64
        Config.batch_size = 10
    
    try:
        # 创建环境（不使用渲染器以专注于物理性能）
        env = OptimizedEnvironment(ground=False, renderer=None)
        
        # 随机添加点
        np.random.seed(42)  # 设置随机种子以确保可重复性
        
        for _ in range(num_points):
            m = np.random.uniform(1, 5)
            pos = np.random.uniform(-50, 50, 3)
            v = np.random.uniform(-5, 5, 3)
            env.add_point(m, pos, v)
        
        # 添加一些弹簧连接
        num_springs = min(num_points * 2, num_points * (num_points - 1) // 2)
        for _ in range(num_springs):
            i = np.random.randint(0, len(env.points))
            j = np.random.randint(0, len(env.points))
            if i != j:
                env.add_spring(env.points[i], env.points[j])
        
        # 运行性能测试
        print(f"Running performance test with {num_points} points and {num_springs} springs")
        print(f"Optimizations: {'enabled' if use_optimizations else 'disabled'}")
        
        start_time = time.time()
        
        # 运行指定步数
        for _ in range(steps):
            env.update_physics()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        steps_per_second = steps / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Completed {steps} steps in {elapsed_time:.2f} seconds")
        print(f"Steps per second: {steps_per_second:.2f}")
        
        # 返回测试结果
        return {
            "num_points": num_points,
            "num_springs": num_springs,
            "steps": steps,
            "elapsed_time": elapsed_time,
            "steps_per_second": steps_per_second,
            "use_optimizations": use_optimizations,
            "config": {
                "precision": str(Config.precision).split('.')[-1],
                "batch_size": Config.batch_size
            }
        }
        
    finally:
        # 恢复原始配置
        Config.precision = original_config["precision"]
        Config.batch_size = original_config["batch_size"]


def simple_pendulum_demo():
    """简单的单摆演示"""
    print("运行简单的单摆演示...")
    
    # 创建环境
    env = Environment(gravity=(0, -9.8, 0), time_step=0.01)
    
    # 添加定点（悬挂点）
    pivot = env.add_ding_point(0, (0, 50, 0), color="red")
    
    # 添加摆锤点
    pendulum = env.add_point(1, (20, 50, 0), r=5, color="blue")
    
    # 添加弹簧连接（作为摆线）
    env.add_spring(pivot, pendulum, k=500)
    
    # 运行模拟
    env.run()


def double_pendulum_demo():
    """双摆演示"""
    print("运行双摆演示...")
    
    # 创建环境
    env = Environment(gravity=(0, -9.8, 0), time_step=0.01)
    
    # 添加定点（悬挂点）
    pivot = env.add_ding_point(0, (200, 0, 200), r=1, color="red")
    
    # 添加摆锤点
    pendulum1 = env.add_point(1, (120, 0, 100), r=1, color="blue")
    pendulum2 = env.add_point(1, (240, 0, 150), r=1, color="green")
    
    # 添加弹簧连接
    env.add_spring(pivot, pendulum1, k=500)
    env.add_spring(pivot, pendulum2, k=500)
    
    # 运行模拟
    env.run()


def creature_demo(creature_type="leg2"):
    """生物模拟演示"""
    print(f"运行{creature_type}生物模拟演示...")
    
    # 运行示例
    run_example(creature_type)


def performance_comparison():
    """性能对比测试"""
    print("运行性能对比测试...")
    
    # 测试不同数量的点
    point_counts = [10, 50, 100, 200]
    
    results = []
    
    for num_points in point_counts:
        print(f"\n测试 {num_points} 个点:")
        
        # 测试优化版本
        optimized_result = run_performance_test(
            num_points=num_points, 
            steps=1000, 
            use_optimizations=True
        )
        
        # 测试非优化版本
        non_optimized_result = run_performance_test(
            num_points=num_points, 
            steps=1000, 
            use_optimizations=False
        )
        results.append((optimized_result, non_optimized_result))

    max_points = max(len(str(item[0]['num_points'])) for item in results)
    max_steps = max(len(str(item[0]['steps'])) for item in results)
    max_time = max(len(f"{item[1]['elapsed_time']:.2f} / {item[0]['elapsed_time']:.2f}") for item in results)
    max_steps_per_sec = max(len(f"{item[1]['steps_per_second']:.2f} / {item[0]['steps_per_second']:.2f}") for item in results)
    max_speedup = len("Speed-up (%)")

    header_points_len = len("Points") + 8
    header_steps_len = len("Steps") + 8
    header_time_len = len("Elapsed Time (s)") + 8
    header_steps_per_sec_len = len("Steps per Second") + 8
    header_speedup_len = len("Speed-up (%)") + 8

    max_points = max(max_points, header_points_len)
    max_steps = max(max_steps, header_steps_len)
    max_time = max(max_time, header_time_len)
    max_steps_per_sec = max(max_steps_per_sec, header_steps_per_sec_len)
    max_speedup = max(max_speedup, header_speedup_len)

    print("\nPerformance Comparison (Non-Optimized / Optimized):")
    print(f"┌{'─' * max_points}┬{'─' * max_steps}┬{'─' * max_time}┬{'─' * max_steps_per_sec}┬{'─' * max_speedup}┐")
    print(f"│ {'Points':<{max_points - 2}} "
          f"│ {'Steps':<{max_steps - 2}} "
          f"│ {'Elapsed Time (s)':<{max_time - 2}} "
          f"│ {'Steps per Second':<{max_steps_per_sec - 2}} "
          f"│ {'Speed-up (%)':<{max_speedup - 2}} │")
    print(f"├{'─' * max_points}┼{'─' * max_steps}┼{'─' * max_time}┼{'─' * max_steps_per_sec}┼{'─' * max_speedup}┤")

    for i, (optimized, non_optimized) in enumerate(results):
        speedup = optimized['steps_per_second'] / non_optimized['steps_per_second'] - 1
        time_str = f"{non_optimized['elapsed_time']:.2f} / {optimized['elapsed_time']:.2f}"
        steps_per_sec_str = f"{non_optimized['steps_per_second']:.2f} / {optimized['steps_per_second']:.2f}"
        print(f"│ {optimized['num_points']:<{max_points - 2}} "
              f"│ {optimized['steps']:<{max_steps - 2}} "
              f"│ {time_str:<{max_time - 2}} "
              f"│ {steps_per_sec_str:<{max_steps_per_sec - 2}} "
              f"│ {speedup:<{max_speedup - 2}.2%} │")

        if i < len(results) - 1:
            print(f"├{'─' * max_points}┼{'─' * max_steps}┼{'─' * max_time}┼{'─' * max_steps_per_sec}┼{'─' * max_speedup}┤")

    print(f"└{'─' * max_points}┴{'─' * max_steps}┴{'─' * max_time}┴{'─' * max_steps_per_sec}┴{'─' * max_speedup}┘")


def custom_creature_demo():
    """自定义生物演示"""
    print("运行自定义生物演示...")
    
    # 创建环境
    env = OptimizedEnvironment(gravity=(0, -9.8, 0), time_step=0.01)
    
    # 创建骨骼
    skeleton = Skeleton(env)
    
    # 添加身体点
    body = skeleton.add_point(5, (0, 20, 0), r=5, color="blue")
    
    # 添加四条腿
    leg_points = []
    for i in range(4):
        # 计算腿部位置
        angle = (i * np.pi / 2) - np.pi / 4
        hip_x = body.pos[0] + np.cos(angle) * 10
        hip_y = body.pos[1] - 5
        
        knee_x = hip_x + np.cos(angle) * 10
        knee_y = hip_y - 15
        
        foot_x = knee_x + np.cos(angle) * 10
        foot_y = knee_y - 15
        
        # 添加腿部点
        hip = skeleton.add_point(1, (hip_x, hip_y, 0))
        knee = skeleton.add_point(1, (knee_x, knee_y, 0))
        foot = skeleton.add_point(2, (foot_x, foot_y, 0), r=2)
        
        leg_points.append((hip, knee, foot))
        
        # 添加骨骼连接
        skeleton.add_spring(body, hip, k=400)
        skeleton.add_spring(hip, knee, k=300)
        skeleton.add_spring(knee, foot, k=300)
        
        # 添加肌肉，不同腿使用不同的相位以创建行走步态
        phase = i * np.pi / 2
        skeleton.add_muscle(body, knee, amp=0.1, freq=0.5, phase=phase, power=150)
        skeleton.add_muscle(hip, foot, amp=0.1, freq=0.5, phase=phase + np.pi/2, power=100)
    
    # 创建生物
    creature = Creature(env, skeleton)
    
    # 运行模拟
    env.run()


def main():
    """主函数"""
    print("优化版牛顿力学物理环境模拟")
    print("=" * 50)
    
    # 显示菜单
    print("请选择演示类型:")
    print("1. 简单单摆")
    print("2. 双摆")
    print("3. 双足生物")
    print("4. 立方体")
    print("5. 类人生物")
    print("6. 昆虫生物")
    print("7. 自定义四足生物")
    print("8. 性能对比测试")
    print("0. 退出")
    
    # 获取用户选择
    choice = input("请输入选择 (0-8): ")
    
    # 根据选择运行相应的演示
    if choice == "1":
        simple_pendulum_demo()
    elif choice == "2":
        double_pendulum_demo()
    elif choice == "3":
        creature_demo("leg2")
    elif choice == "4":
        creature_demo("box")
    elif choice == "5":
        creature_demo("humanb")
    elif choice == "6":
        creature_demo("insect")
    elif choice == "7":
        custom_creature_demo()
    elif choice == "8":
        performance_comparison()
    elif choice == "0":
        print("谢谢使用，再见！")
        sys.exit(0)
    else:
        print("无效的选择，请重新运行程序。")
        sys.exit(1)
