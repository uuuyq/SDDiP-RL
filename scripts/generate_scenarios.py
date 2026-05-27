"""生成多个 scenario_data 文件"""
from pathlib import Path

import pandas as pd

from sddip.sddip import scenarios


def generate_scenario_files(
    output_dir: Path,
    base_case_dir: Path,
    n_stages: int = 24,
    n_realizations_per_stage: int = 6,
    n_files: int = 10,
) -> None:
    """生成多个 scenario_data 文件

    Args:
        output_dir: 输出目录
        base_case_dir: 基础案例目录（包含 bus_data.txt 和 ren_data.txt）
        n_stages: 阶段数
        n_realizations_per_stage: 每个阶段的实现数
        n_files: 生成的文件数量
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    bus_file_path = base_case_dir / "bus_data.txt"
    renewables_file_path = base_case_dir / "ren_data.txt"

    bus_df = pd.read_csv(bus_file_path, delimiter=r"\s+")
    renewables_df = pd.read_csv(renewables_file_path, delimiter=r"\s+")

    demands = bus_df["Pd"].values.tolist()
    re_max_frac = renewables_df["max_frac"].values.tolist()

    n_buses = len(demands)
    demand_buses = [b for b in range(n_buses) if demands[b] != 0]
    max_demand_value_targets = [2 * d for d in demands if d != 0]
    renewables_buses = [b for b in range(n_buses) if re_max_frac[b] != 0]

    # Renewables 参数
    min_values = [0] * len(renewables_buses)
    max_values = [frac * sum(demands) for frac in re_max_frac if frac != 0]
    start_values = [0.1 * m for m in max_values]
    step_sizes = [1 / 3 * m for m in max_values]

    sc_generator = scenarios.ScenarioGenerator(n_stages, n_realizations_per_stage)

    for i in range(n_files):
        # Demand scenarios
        demand_scenario_df = sc_generator.generate_demand_scenario_dataframe(
            n_buses, demand_buses, max_demand_value_targets, 0.2
        )

        # Renewables scenarios
        renewables_scenario_df = sc_generator.generate_renewables_scenario_dataframe(
            n_buses,
            renewables_buses,
            start_values,
            step_sizes,
            min_values,
            max_values,
            0.3,
            0.2,
        )
        renewables_scenario_df = renewables_scenario_df.drop(["t", "n", "p"], axis=1)

        scenario_df = pd.concat([demand_scenario_df, renewables_scenario_df], axis=1)

        # 保存到文件，使用 \t 分隔
        output_path = output_dir / f"scenario_{i + 1}.txt"
        scenario_df.to_csv(output_path, sep="\t", index=False)
        print(f"已生成: {output_path}")


if __name__ == "__main__":
    # 使用 case6ww t24_n06 的数据作为基础
    base_case = Path(r"d:\tools\workspace_pycharm\SDDiP-RL\data\01_test_cases\case6ww\t24_n06")
    output = Path(r"d:\tools\workspace_pycharm\SDDiP-RL\bundle_ml\scenario_data")

    generate_scenario_files(output, base_case, n_stages=24, n_realizations_per_stage=6, n_files=50)