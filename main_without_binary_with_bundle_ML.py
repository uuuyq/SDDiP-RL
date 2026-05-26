from pathlib import Path
from sddip.sddip import dualsolver
from sddip.sddip.sddipclassical_without_binary_with_bundle_ML import Algorithm


def main():

    path = Path(r".\data\01_test_cases\case6ww\t24_n06")
    log_path = r".\log"

    # dual_solver_stop_tolerance: float = field(default=10 ** -6)
    #     dual_solver_time_limit: int = field(default=5 * 60)
    #     dual_solver_max_iterations: int = field(default=5000)
    dual_solver = dualsolver.BundleMethod(
        5000,
        10 ** -6,
        log_path,
        predicted_ascent="abs",
        time_limit=5 * 60,
    )
    mylog_dir = "lag_without_binary_bundle_ML.log"

    algorithm = Algorithm(path, log_path, dual_solver, mylog_dir, instance_name="instance_01")
    algorithm.run(10)


if __name__ == "__main__":
    main()





