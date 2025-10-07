#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""物理模拟环境性能测试示例"""

import numpy as np
import time
from optimized_engine import Point, Config
from optimized_env import PhysicsEnv, make_env
from optimized_renderer import Renderer
from typing import Dict, Tuple


class PerformanceTester:
    """物理模拟性能测试器"""
    def __init__(self):
        self.results = {}
    
    def test_computation(self, num_points: int = 100, steps: int = 1000) -> Dict[str, float]:
        """
        测试计算性能
        
        :param num_points: 点的数量
        :param steps: 模拟步数
        :return: 性能测试结果
        """
        print(f"\n=== 计算性能测试: {num_points} 个点, {steps} 步 ===")
        
        # 清除所有点
        Point.clear()
        
        # 创建点
        points = []
        for i in range(num_points):
            pos = np.random.uniform(-100, 100, 3)
            v = np.random.uniform(-10, 10, 3)
            points.append(Point(1, pos, v, color="blue"))
        
        # 创建弹簧连接
        from optimized_walker import Skeleton
        springs = []
        for i in range(num_points - 1):
            springs.append(Skeleton(points[i], points[i+1], k=50))
        
        # 创建生物
        from optimized_walker import Creature
        creature = Creature(points, [], springs)
        
        # 测量计算时间
        print("开始计算...")
        start_time = time.time()
        
        for _ in range(steps):
            # 运行生物物理
            creature.run()
            # 应用重力
            Point.gravity()
            # 更新物理状态
            Point.run1(0.01)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 计算性能指标
        avg_step_time = (elapsed / steps) * 1000  # 毫秒/步
        steps_per_second = steps / elapsed  # 步/秒
        
        # 存储结果
        result = {
            'num_points': num_points,
            'steps': steps,
            'total_time': elapsed,
            'avg_step_time': avg_step_time,
            'steps_per_second': steps_per_second
        }
        
        # 打印结果
        print(f"总耗时: {elapsed:.4f} 秒")
        print(f"每步平均耗时: {avg_step_time:.4f} 毫秒")
        print(f"每秒可处理: {steps_per_second:.2f} 步")
        
        return result
    
    def test_rendering(self, num_points: int = 100, duration: float = 5.0) -> Dict[str, float]:
        """
        测试渲染性能
        
        :param num_points: 点的数量
        :param duration: 测试持续时间（秒）
        :return: 性能测试结果
        """
        print(f"\n=== 渲染性能测试: {num_points} 个点, 持续 {duration} 秒 ===")
        
        # 清除所有点
        Point.clear()
        
        # 创建点
        points = []
        for i in range(num_points):
            pos = np.random.uniform(-100, 100, 3)
            v = np.random.uniform(-5, 5, 3)
            points.append(Point(1, pos, v, color=self._get_random_color()))
        
        # 创建弹簧连接
        from optimized_walker import Skeleton
        springs = []
        for i in range(num_points - 1):
            springs.append(Skeleton(points[i], points[i+1], k=50))
        
        # 创建生物
        from optimized_walker import Creature
        creature = Creature(points, [], springs)
        
        # 创建渲染器
        renderer = Renderer(width=800, height=600, title=f"渲染性能测试: {num_points} 个点")
        
        print("开始渲染测试...")
        print("按ESC键退出测试")
        
        # 测试渲染性能
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration and renderer.is_running():
            # 处理事件
            renderer.handle_events()
            
            # 运行物理模拟
            creature.run()
            Point.gravity()
            Point.run1(0.01)
            
            # 渲染画面
            renderer.render()
            frame_count += 1
            
            # 控制帧率
            renderer.tick(1000)  # 不限制帧率，测试最大性能
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 计算性能指标
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # 存储结果
        result = {
            'num_points': num_points,
            'duration': elapsed,
            'frames': frame_count,
            'fps': fps
        }
        
        # 打印结果
        print(f"实际持续时间: {elapsed:.2f} 秒")
        print(f"总帧数: {frame_count}")
        print(f"平均FPS: {fps:.2f}")
        
        # 关闭渲染器
        renderer.close()
        
        return result
    
    def _get_random_color(self) -> Tuple[int, int, int]:
        """生成随机颜色"""
        return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    
    def run_all_tests(self):
        """运行所有性能测试"""
        print("开始所有性能测试...")
        
        # 测试不同规模的计算性能
        for num_points in [10, 50, 100, 200]:
            result = self.test_computation(num_points=num_points)
            self.results[f"computation_{num_points}"] = result
        
        # 测试不同规模的渲染性能
        for num_points in [10, 50, 100]:
            result = self.test_rendering(num_points=num_points, duration=3.0)
            self.results[f"rendering_{num_points}"] = result
        
        # 显示汇总结果
        self._show_summary()
    
    def _show_summary(self):
        """显示测试汇总结果"""
        print("\n=== 性能测试汇总 ===")
        
        # 显示计算性能汇总
        print("\n计算性能:")
        print("----------------------------------------")
        print("点数    总耗时(秒)  每步耗时(毫秒)  每秒步数")
        print("----------------------------------------")
        
        for key in sorted(self.results.keys()):
            if key.startswith("computation_"):
                res = self.results[key]
                print(f"{res['num_points']:<6} {res['total_time']:<10.4f} {res['avg_step_time']:<14.4f} {res['steps_per_second']:<.2f}")
        
        # 显示渲染性能汇总
        print("\n渲染性能:")
        print("----------------------------------------")
        print("点数    持续时间(秒)  总帧数   平均FPS")
        print("----------------------------------------")
        
        for key in sorted(self.results.keys()):
            if key.startswith("rendering_"):
                res = self.results[key]
                print(f"{res['num_points']:<6} {res['duration']:<12.2f} {res['frames']:<7} {res['fps']:<.2f}")

# 演示如何使用优化后的物理环境
def demo_environment():
    """演示如何使用优化后的物理环境"""
    print("\n=== 物理环境使用示例 ===")
    
    # 创建环境
    env = make_env('Balance-v0', in3d=False)
    
    print("环境信息:")
    print(f"动作空间: {env.get_action_space()}")
    print(f"观察空间: {env.get_observation_space()}")
    
    # 重置环境
    obs = env.reset()
    print(f"初始观察维度: {len(obs)}")
    
    # 运行环境模拟
    print("\n开始环境模拟...")
    print("控制说明:")
    print("  ESC - 退出")
    print("  F - 显示/隐藏FPS")
    print("  A - 显示/隐藏加速度向量")
    print("  V - 显示/隐藏速度向量")
    print("  WASD/空格/左Ctrl - 移动相机")
    print("  方向键 - 旋转相机")
    
    total_reward = 0
    steps = 0
    
    # 创建一个简单的控制器来测试环境
    try:
        while True:
            # 随机动作
            action = np.random.uniform(-1.0, 1.0, size=len(env.creature.muscles))
            
            # 执行一步
            obs, reward, done, info = env.step(action)
            
            # 渲染环境
            env.render()
            
            # 累积奖励
            total_reward += reward
            steps += 1
            
            # 打印信息
            if steps % 10 == 0:
                print(f"步数: {steps}, 总奖励: {total_reward:.2f}, 质心Y: {info['centroid_position'][1]:.2f}")
            
            # 检查是否终止
            if done:
                print(f"\n模拟结束 - 步数: {steps}, 总奖励: {total_reward:.2f}")
                break
    except KeyboardInterrupt:
        print("\n用户中断模拟")
    finally:
        # 关闭环境
        env.close()

def main():
    """主函数"""
    print("欢迎使用优化后的物理模拟环境！")
    print("这个环境使用Numpy和Pygame进行了性能优化，适用于大规模物理模拟。")
    
    while True:
        print("\n请选择操作：")
        print("1. 运行性能测试")
        print("2. 运行环境示例")
        print("3. 退出")
        
        try:
            choice = int(input("请输入选项 (1-3): "))
            
            if choice == 1:
                tester = PerformanceTester()
                tester.run_all_tests()
            elif choice == 2:
                demo_environment()
            elif choice == 3:
                print("谢谢使用，再见！")
                break
            else:
                print("无效的选项，请重新输入")
        except ValueError:
            print("请输入有效的数字")

if __name__ == "__main__":
    main()