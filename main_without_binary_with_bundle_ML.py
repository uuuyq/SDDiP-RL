from pathlib import Path
from multiprocessing import Pool, cpu_count
import os

from sddip.sddip import dualsolver
from sddip.sddip.sddipclassical_without_binary_with_bundle_ML import Algorithm


def run_scenario(args):
    """运行单个 scenario 的求解（独立进程）"""
    scenario_path, scenario_id, base_path, log_path = args

    # 每个 scenario 使用独立的 log 目录
    scenario_log_dir = Path(log_path) / f"scenario_{scenario_id}"
    scenario_log_dir.mkdir(exist_ok=True, parents=True)

    # 日志文件放在 scenario 对应目录下
    mylog_dir = scenario_log_dir / f"lag_without_binary_bundle_ML_scenario_{scenario_id}.log"

    # 创建 dual_solver
    dual_solver = dualsolver.BundleMethod(
        5000,
        10 ** -6,
        str(scenario_log_dir),
        predicted_ascent="abs",
        time_limit=5 * 60,
    )

    # 创建 algorithm 实例，使用独立的 scenario 文件
    algorithm = Algorithm(
        base_path,
        str(scenario_log_dir),
        dual_solver,
        str(mylog_dir),  # 传递完整路径
        instance_name=f"scenario_{scenario_id}",
        scenario_path=scenario_path,
    )

    print(f"[PID {os.getpid()}] Starting scenario_{scenario_id}")
    algorithm.run(10)
    print(f"[PID {os.getpid()}] Finished scenario_{scenario_id}")

    return scenario_id


def main():
    # 基础路径配置
    base_path = Path(r".\data\01_test_cases\case6ww\t24_n06")
    scenario_data_dir = Path(r".\bundle_ml\scenario_data")
    log_path = r".\log"

    # 自动检测 CPU 核心数，但限制最大为 4
    max_workers = min(cpu_count(), 4)
    print(f"Available CPU cores: {cpu_count()}, using {max_workers} workers")

    # 收集所有 scenario 文件
    scenario_files = []
    for i in range(1, 51):  # scenario_1.txt 到 scenario_50.txt
        scenario_file = scenario_data_dir / f"scenario_{i}.txt"
        if scenario_file.exists():
            scenario_files.append((str(scenario_file), i, base_path, log_path))
        else:
            print(f"Warning: {scenario_file} does not exist, skipping")

    print(f"Found {len(scenario_files)} scenario files")
    print(f"Running with {max_workers} parallel processes")

    # 多进程并行执行
    with Pool(processes=max_workers) as pool:
        results = pool.map(run_scenario, scenario_files)

    print(f"All {len(results)} scenarios completed!")


if __name__ == "__main__":
    main()
