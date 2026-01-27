from sddip.sddip.sddipclassical import Algorithm
# from sddip.sddip.sddipclassical_without_binary_normalized import Algorithm
# from sddip.sddip.sddipclassical_normalized import Algorithm
from sddip.sddip import dualsolver
from pathlib import Path

# @dataclass
# class TestSetup:
#     name: str
#     path: Path
#     algorithm: Literal["sddip", "dsddip"]
#
#     sddip_n_binaries: int = field(default=5)
#
#     sddip_max_iterations: int = field(default=100)
#     sddip_time_limit: int = field(default=5 * 60)
#
#     sddip_refinment_stabilization_count: int = field(default=5)
#     sddip_stop_stabilization_count: int = field(default=1000)
#
#     sddip_no_improvement_tolerance: float = field(default=10**-6)
#
#     sddip_primary_cut_type: str = field(default="sb")
#     sddip_n_samples_primary: int = field(default=3)
#     sddip_secondary_cut_type: str = field(default="l")
#     sddip_n_samples_secondary: int = field(default=1)
#
#     sddip_projection_big_m: float = field(default=10**4)
#
#     sddip_n_samples_final_ub: int = field(default=300)
#
#     dual_solver_stop_tolerance: float = field(default=10**-6)
#     dual_solver_time_limit: int = field(default=5 * 60)
#     dual_solver_max_iterations: int = field(default=5000)
#
#     seed: Seed = field(default_factory=lambda: int(time.time()))
#
#     @classmethod
#     def from_dict(cls, d: dict[str, Any], /) -> "TestSetup":
#         """Create a `TestSetup` object from a dictionary."""
#         d["path"] = Path(d["path"])
#         return cls(**d)
#
#
# Setup = list[TestSetup]



def main():

    path = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t06_n06")
    log_path = r"D:\tools\workspace_pycharm\sddip-main-zou\log"

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
    mylog_dir = "lag_binary.log"

    algorithm = Algorithm(path, log_path, dual_solver, mylog_dir)
    algorithm.run(10)


if __name__ == "__main__":
    main()





