"""
测试使用 bundle_fast 加速的 SDDiP 算法
"""
from sddip.sddip.sddipclassical_without_binary_with_bundle_fast import Algorithm
from sddip.sddip import dualsolver
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_timing_results(timing_file: str, output_file: str = "sddip_timing_analysis.png"):
    """
    绘制SDDiP时间和边界分析图
    
    Args:
        timing_file: 时间记录JSON文件路径
        output_file: 输出图像文件路径
    """
    # 读取数据
    with open(timing_file, 'r', encoding='utf-8') as f:
        timing_data = json.load(f)
    
    iterations = [d['iteration'] for d in timing_data]
    
    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('SDDiP with Bundle Fast - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. 前向 vs 后向时间
    ax1 = axes[0, 0]
    forward_times = [d['forward_pass']['duration_seconds'] for d in timing_data]
    backward_times = [d['backward_pass']['duration_seconds'] for d in timing_data]
    
    x = np.arange(len(iterations))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, forward_times, width, label='Forward Pass', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(x + width/2, backward_times, width, label='Backward Pass', alpha=0.7, color='salmon')
    
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Time (seconds)', fontsize=11)
    ax1.set_title('Forward vs Backward Pass Time', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(iterations)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 2. 收敛边界
    ax2 = axes[0, 1]
    lower_bounds = [d['bounds']['lower_bound'] for d in timing_data]
    upper_bounds = [d['bounds']['upper_bound_l'] for d in timing_data]
    gaps = [d['bounds']['gap'] for d in timing_data if d['bounds']['gap'] is not None]
    
    ax2.plot(iterations, lower_bounds, 'b-o', label='Lower Bound', linewidth=2, markersize=6)
    ax2.plot(iterations, upper_bounds, 'r-s', label='Upper Bound', linewidth=2, markersize=6)
    ax2.fill_between(iterations, lower_bounds, upper_bounds, alpha=0.2, color='green')
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Objective Value', fontsize=11)
    ax2.set_title('Convergence Bounds', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. n=0 vs n>0 Bundle时间对比
    ax3 = axes[1, 0]
    n_0_times = []
    n_gt_0_times = []
    
    for d in timing_data:
        bundle_calls = d['backward_pass'].get('bundle_calls', {})
        total_n_0 = 0
        total_n_gt_0 = 0
        
        for stage_key, stage_data in bundle_calls.items():
            # n=0 时间
            for res_key, res_data in stage_data.get('n_0_exact', {}).items():
                total_n_0 += res_data['duration_seconds']
            # n>0 时间
            for res_key, res_data in stage_data.get('n_gt_0_approx', {}).items():
                total_n_gt_0 += res_data['duration_seconds']
        
        n_0_times.append(total_n_0)
        n_gt_0_times.append(total_n_gt_0)
    
    bars3 = ax3.bar(x - width/2, n_0_times, width, label='n=0 (Exact)', alpha=0.7, color='lightgreen')
    bars4 = ax3.bar(x + width/2, n_gt_0_times, width, label='n>0 (Approx)', alpha=0.7, color='orange')
    
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Time (seconds)', fontsize=11)
    ax3.set_title('Bundle Method: n=0 vs n>0 Time', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(iterations)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Gap收敛
    ax4 = axes[1, 1]
    ax4.plot(iterations, gaps, 'g-^', linewidth=2, markersize=8)
    ax4.fill_between(iterations, gaps, alpha=0.3, color='green')
    
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Gap', fontsize=11)
    ax4.set_title('Optimality Gap Convergence', fontsize=12, fontweight='bold')
    ax4.legend(['Gap'])
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Bundle迭代次数分布
    ax5 = axes[2, 0]
    n_0_iters = []
    n_gt_0_iters = []
    
    for d in timing_data:
        bundle_calls = d['backward_pass'].get('bundle_calls', {})
        iters_n_0 = []
        iters_n_gt_0 = []
        
        for stage_key, stage_data in bundle_calls.items():
            for res_key, res_data in stage_data.get('n_0_exact', {}).items():
                iters_n_0.append(res_data['iterations'])
            for res_key, res_data in stage_data.get('n_gt_0_approx', {}).items():
                iters_n_gt_0.append(res_data['iterations'])
        
        n_0_iters.append(np.mean(iters_n_0) if iters_n_0 else 0)
        n_gt_0_iters.append(np.mean(iters_n_gt_0) if iters_n_gt_0 else 0)
    
    bars5 = ax5.bar(x - width/2, n_0_iters, width, label='n=0 (Exact)', alpha=0.7, color='lightcoral')
    bars6 = ax5.bar(x + width/2, n_gt_0_iters, width, label='n>0 (Approx)', alpha=0.7, color='lightblue')
    
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Average Iterations', fontsize=11)
    ax5.set_title('Bundle Method Iterations', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(iterations)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 总时间累积
    ax6 = axes[2, 1]
    cumulative_forward = np.cumsum(forward_times)
    cumulative_backward = np.cumsum(backward_times)
    cumulative_total = cumulative_forward + cumulative_backward
    
    ax6.plot(iterations, cumulative_forward, 'b-o', label='Cumulative Forward', linewidth=2)
    ax6.plot(iterations, cumulative_backward, 'r-s', label='Cumulative Backward', linewidth=2)
    ax6.plot(iterations, cumulative_total, 'g-^', label='Cumulative Total', linewidth=2, markersize=8)
    
    ax6.set_xlabel('Iteration', fontsize=11)
    ax6.set_ylabel('Cumulative Time (seconds)', fontsize=11)
    ax6.set_title('Cumulative Time Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Timing analysis plot saved to: {output_file}")
    plt.close()


def main():
    # 数据路径
    path = Path(r"D:\tools\workspace_pycharm\SDDiP-RL\data\01_test_cases\case6ww\t06_n06")
    log_path = r"D:\tools\workspace_pycharm\SDDiP-RL\log"

    # 配置 Bundle Method 对偶求解器
    # 参数: max_iterations, tolerance, log_dir, predicted_ascent, time_limit
    dual_solver = dualsolver.BundleMethod(
        max_iterations=5000,
        tolerance=10 ** -6,
        log_dir=log_path,
        predicted_ascent="abs",
        time_limit=5 * 60,  # 5分钟
    )
    
    # 日志文件
    mylog_dir = "lag_without_binary_with_bundle_fast.log"

    # 创建算法实例
    algorithm = Algorithm(path, log_path, dual_solver, mylog_dir)
    
    # 运行算法（10次迭代）
    algorithm.run(10)
    
    # 绘制时间和收敛分析图
    timing_file = str(algorithm.timing_log_path)
    print(f"\nGenerating timing analysis plot from: {timing_file}")
    plot_timing_results(timing_file, output_file="sddip_timing_analysis.png")


if __name__ == "__main__":
    main()
