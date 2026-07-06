"""
氢气负荷模拟 - 复现 Zhao et al. (2023) Section III
A Hydrogen Load Modeling Method for Integrated Hydrogen Energy System Planning

模拟加氢站一天内的氢气负荷曲线
基于离散事件仿真方法
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ============ 参数设置 (来自文献 Table I 和 Section III) ============
H2_TANK_CAPACITY = 33.0       # 氢燃料卡车罐容量 (kg)
NUM_DISPENSERS = 6            # 加氢位数量
MEAN_INTERARRIVAL = 5.0       # 卡车到达间隔均值 (分钟), 指数分布
MEAN_FUELING_TIME = 5.5       # 加氢时间均值 (分钟), 正态分布
STD_FUELING_TIME = 0.83       # 加氢时间标准差 (分钟)
WORK_START = 9 * 60           # 工作开始时间 (分钟) 9:00
WORK_END = 18 * 60            # 工作结束时间 (分钟) 18:00
SIM_DURATION = WORK_END - WORK_START  # 仿真时长 540分钟

np.random.seed(42)


def simulate_hydrogen_load():
    """
    离散事件仿真：模拟加氢站一天的氢气负荷

    Returns:
        time_points: 时间点列表 (分钟，从0点开始计)
        load_profile: 对应的氢气负荷 (kg/h)
        truck_count_profile: 对应的在站卡车数
    """
    # 仿真状态
    clock = 0.0                    # 仿真时钟 (分钟，从9:00开始)
    dispensers = []                # 正在服务的加氢位: [(departure_time, fueling_time, truck_id)]
    queue = deque()                # 等待队列: [(arrival_time, truck_id)]
    truck_id = 0

    # 记录事件
    events = []  # [(time, load_change_kg_per_min)]

    # 生成第一辆卡车到达时间
    first_arrival = np.random.exponential(MEAN_INTERARRIVAL)
    next_arrival = first_arrival
    truck_id += 1

    # 仿真主循环
    while clock < SIM_DURATION:
        # 确定下一个事件：到达 or 离开
        next_departure = min([d[0] for d in dispensers], default=float('inf'))

        if next_arrival <= next_departure:
            # === 到达事件 ===
            clock = next_arrival

            if clock > SIM_DURATION:
                break

            # 记录到达前的负荷变化（如果有离开的话在离开事件处理）
            # 检查是否有空闲加氢位
            if len(dispensers) < NUM_DISPENSERS:
                # 直接开始加氢
                fueling_time = np.random.normal(MEAN_FUELING_TIME, STD_FUELING_TIME)
                fueling_time = max(3.0, min(8.0, fueling_time))  # 限制在3-8分钟范围
                departure_time = clock + fueling_time
                dispensers.append((departure_time, fueling_time, truck_id))

                # 该卡车对氢气负荷的贡献: 33/fueling_time kg/min
                load_rate = H2_TANK_CAPACITY / fueling_time  # kg/min
                events.append((clock, load_rate, 'arrive', truck_id))
            else:
                # 加入等待队列
                queue.append((clock, truck_id))

            # 生成下一辆卡车到达时间
            next_arrival = clock + np.random.exponential(MEAN_INTERARRIVAL)
            truck_id += 1

        else:
            # === 离开事件 ===
            # 找到最早离开的加氢位
            min_idx = np.argmin([d[0] for d in dispensers])
            dep_time, fueling_time, dep_truck_id = dispensers.pop(min_idx)
            clock = dep_time

            if clock > SIM_DURATION:
                # 超过工作时间，该卡车实际加氢时间缩短
                actual_fueling_time = SIM_DURATION - (clock - fueling_time)
                # 重新计算：该卡车在关门前已加氢的时间对应的负荷
                # 离开事件中，需要减去该卡车的负荷贡献
                # 但由于超过6pm后不再加氢，负荷贡献按实际加氢时间计算
                # 原始贡献率: 33/fueling_time kg/min
                # 实际加氢时间: actual_fueling_time 分钟
                # 实际消耗氢气: 33/fueling_time * actual_fueling_time kg
                # 但在仿真中我们记录的是负荷率，所以需要先减去原始负荷率
                load_rate = H2_TANK_CAPACITY / fueling_time
                events.append((SIM_DURATION, load_rate, 'depart_late', dep_truck_id))
                # 不再处理队列中的卡车（已过工作时间）
                continue

            # 该卡车离开，减去其负荷贡献
            load_rate = H2_TANK_CAPACITY / fueling_time
            events.append((clock, load_rate, 'depart', dep_truck_id))

            # 检查队列中是否有等待的卡车
            if queue and clock < SIM_DURATION:
                arr_time, wait_truck_id = queue.popleft()
                fueling_time_new = np.random.normal(MEAN_FUELING_TIME, STD_FUELING_TIME)
                fueling_time_new = max(3.0, min(8.0, fueling_time_new))
                departure_time_new = clock + fueling_time_new
                dispensers.append((departure_time_new, fueling_time_new, wait_truck_id))

                load_rate_new = H2_TANK_CAPACITY / fueling_time_new
                events.append((clock, load_rate_new, 'arrive_from_queue', wait_truck_id))

    # 处理仍在加氢的卡车（超过6pm的部分）
    for dep_time, fueling_time, t_id in dispensers:
        start_time = dep_time - fueling_time
        if start_time < SIM_DURATION:
            # 该卡车在6pm前开始加氢，6pm后停止
            load_rate = H2_TANK_CAPACITY / fueling_time
            events.append((SIM_DURATION, load_rate, 'depart_late', t_id))

    # ============ 构建负荷曲线 ============
    # 按时间排序事件
    events.sort(key=lambda x: x[0])

    # 计算每个时间点的负荷
    time_resolution = 1  # 分钟
    time_points_min = []  # 从0点开始的分钟数
    load_kg_per_min = []  # kg/min
    truck_count = []

    current_load = 0.0  # 当前负荷 kg/min
    current_trucks = 0  # 当前在站卡车数
    event_idx = 0

    for t in range(0, SIM_DURATION + 1, time_resolution):
        # 处理在时间t发生的所有事件
        while event_idx < len(events) and events[event_idx][0] <= t:
            _, load_rate, event_type, _ = events[event_idx]
            if event_type in ['arrive', 'arrive_from_queue']:
                current_load += load_rate
                current_trucks += 1
            elif event_type in ['depart', 'depart_late']:
                current_load -= load_rate
                current_trucks -= 1
            event_idx += 1

        # 映射到0点开始的时间
        actual_time = WORK_START + t
        time_points_min.append(actual_time)
        load_kg_per_min.append(max(0, current_load))
        truck_count.append(max(0, current_trucks))

    # 转换负荷单位: kg/min -> kg/h
    load_kg_per_h = [l * 60 for l in load_kg_per_min]

    # 扩展到24小时（9:00前和18:00后负荷为0）
    full_time = list(range(0, 24 * 60 + 1, time_resolution))
    full_load = []
    full_truck_count = []
    for t in full_time:
        if WORK_START <= t <= WORK_END:
            idx = t - WORK_START
            if idx < len(load_kg_per_h):
                full_load.append(load_kg_per_h[idx])
                full_truck_count.append(truck_count[idx])
            else:
                full_load.append(0)
                full_truck_count.append(0)
        else:
            full_load.append(0)
            full_truck_count.append(0)

    return full_time, full_load, full_truck_count


def plot_hydrogen_load(time_min, load_kg_h, truck_count):
    """绘制24小时氢气负荷曲线和在站卡车数"""
    # 时间轴转换为小时
    time_h = [t / 60 for t in time_min]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # (a) 氢气负荷曲线
    ax1.fill_between(time_h, load_kg_h, alpha=0.3, color='#2196F3')
    ax1.plot(time_h, load_kg_h, color='#1565C0', linewidth=1.5)
    ax1.set_ylabel('Hydrogen Load (kg/h)', fontsize=13)
    ax1.set_title('Hydrogen Load Profile on a Typical Day', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 1))
    ax1.grid(True, alpha=0.3)

    # 标注峰值
    peak_load = max(load_kg_h)
    peak_idx = load_kg_h.index(peak_load)
    peak_time = time_h[peak_idx]
    ax1.annotate(f'Peak: {peak_load:.2f} kg/h\nat {peak_time:.1f}h',
                 xy=(peak_time, peak_load),
                 xytext=(peak_time + 1.5, peak_load * 0.85),
                 fontsize=11,
                 arrowprops=dict(arrowstyle='->', color='red'),
                 color='red', fontweight='bold')

    # 标注工作时间
    ax1.axvspan(9, 18, alpha=0.05, color='green', label='Working hours (9:00-18:00)')
    ax1.legend(fontsize=10, loc='upper left')

    # (b) 在站卡车数
    ax2.fill_between(time_h, truck_count, alpha=0.3, color='#FF9800')
    ax2.plot(time_h, truck_count, color='#E65100', linewidth=1.5)
    ax2.set_ylabel('Number of Trucks in Station', fontsize=13)
    ax2.set_xlabel('Time of Day (h)', fontsize=13)
    ax2.set_title('Number of Hydrogen-Fueled Trucks in the Station', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 1))
    ax2.set_yticks(range(0, max(truck_count) + 2))
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(9, 18, alpha=0.05, color='green')

    # 标注最大卡车数
    max_trucks = max(truck_count)
    max_idx = truck_count.index(max_trucks)
    max_time = time_h[max_idx]
    ax2.annotate(f'Max: {max_trucks} trucks\nat {max_time:.1f}h',
                 xy=(max_time, max_trucks),
                 xytext=(max_time + 1.5, max_trucks * 0.7),
                 fontsize=11,
                 arrowprops=dict(arrowstyle='->', color='red'),
                 color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(r'd:\tools\workspace_pycharm\SDDiP-RL\modeling\H2_model\hydrogen_load_profile.png',
                dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\n图片已保存至: d:\\tools\\workspace_pycharm\\SDDiP-RL\\modeling\\H2_model\\hydrogen_load_profile.png")


if __name__ == '__main__':
    print("=" * 60)
    print("氢气负荷仿真 - Zhao et al. (2023)")
    print("A Hydrogen Load Modeling Method for Integrated")
    print("Hydrogen Energy System Planning")
    print("=" * 60)
    print(f"\n仿真参数:")
    print(f"  加氢站工作时间: 9:00 - 18:00")
    print(f"  卡车H2罐容量: {H2_TANK_CAPACITY} kg")
    print(f"  加氢位数量: {NUM_DISPENSERS}")
    print(f"  到达间隔: 指数分布(均值={MEAN_INTERARRIVAL} min)")
    print(f"  加氢时间: 正态分布(均值={MEAN_FUELING_TIME} min, 标准差={STD_FUELING_TIME} min)")
    print(f"\n开始仿真...")

    time_min, load_kg_h, truck_count = simulate_hydrogen_load()

    # 统计信息
    working_load = [l for t, l in zip(time_min, load_kg_h) if WORK_START <= t <= WORK_END]
    total_h2 = sum(l / 60 for l in working_load)  # kg (对1分钟间隔求和再除以60)
    num_trucks_served = max(truck_count)  # 近似

    print(f"\n仿真结果:")
    print(f"  峰值氢气负荷: {max(load_kg_h):.2f} kg/h")
    print(f"  最大在站卡车数: {max(truck_count)}")
    print(f"  工作时间内总氢气消耗: {total_h2:.2f} kg")
    print(f"  工作时间内平均氢气负荷: {np.mean(working_load):.2f} kg/h")

    plot_hydrogen_load(time_min, load_kg_h, truck_count)
