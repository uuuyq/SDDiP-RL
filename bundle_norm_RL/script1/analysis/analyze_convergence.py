"""分析不收敛 config 的原因"""
import os, sys, glob
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bundle_norm_RL.script.config import LevelBundleConfig
from bundle_norm_RL.script.logger import get_logger
from bundle_norm_RL.script.level_bundle_problem import LevelBundleSolver


def main():
    log = get_logger(os.path.join(os.path.dirname(__file__), "convergence_analysis.log"))
    configs_dir = r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_norm_RL\configs"
    pkl_files = sorted(glob.glob(os.path.join(configs_dir, "config_*.pkl")))

    reasons = {
        "gap_converged_but_pi0_too_small": 0,
        "gap_converged_but_LB_too_small": 0,
        "gap_not_converged": 0,
        "converged": 0,
    }

    not_converged_details = []

    for idx, pkl_path in enumerate(pkl_files):
        basename = os.path.basename(pkl_path)
        parts = basename.replace("config_", "").replace(".pkl", "").split("_")
        iteration, t, n = int(parts[0]), int(parts[1]), int(parts[2])

        try:
            config = LevelBundleConfig.from_pkl(pkl_path)
        except Exception:
            continue

        solver = LevelBundleSolver(log, config, n=n)
        results = solver.solve()

        if results.converged:
            reasons["converged"] += 1
        else:
            if results.ub is not None and results.lb is not None:
                gap = results.ub - results.lb
                rel_gap = gap / max(abs(results.ub), 1e-10)

                if rel_gap < config.gap_tol or gap < 1e-6:
                    if results.pi0_star is not None and results.pi0_star < 1e-6:
                        reasons["gap_converged_but_pi0_too_small"] += 1
                        not_converged_details.append({
                            "config": basename, "i": iteration, "t": t, "n": n,
                            "reason": "pi0_too_small",
                            "pi0_star": results.pi0_star,
                            "LB": results.lb, "UB": results.ub,
                        })
                    else:
                        reasons["gap_converged_but_LB_too_small"] += 1
                        not_converged_details.append({
                            "config": basename, "i": iteration, "t": t, "n": n,
                            "reason": "LB_pi0_ratio_too_small",
                            "pi0_star": results.pi0_star,
                            "LB": results.lb, "UB": results.ub,
                        })
                else:
                    reasons["gap_not_converged"] += 1
                    not_converged_details.append({
                        "config": basename, "i": iteration, "t": t, "n": n,
                        "reason": "gap_not_converged",
                        "pi0_star": results.pi0_star,
                        "LB": results.lb, "UB": results.ub,
                        "rel_gap": rel_gap,
                        "n_iters": results.n_iterations,
                    })
            else:
                reasons["gap_not_converged"] += 1

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(pkl_files)}] processed...")

    total = sum(reasons.values())
    conv_rate = reasons["converged"] / total * 100

    print("=" * 60)
    print("Convergence Analysis (Original Solver)")
    print("=" * 60)
    for k, v in reasons.items():
        print(f"  {k}: {v}")
    print(f"\nTotal: {total}")
    print(f"Convergence rate: {conv_rate:.1f}%")

    # gap_not_converged 分析
    gap_nc = [d for d in not_converged_details if d["reason"] == "gap_not_converged"]
    if gap_nc:
        rel_gaps = [d["rel_gap"] for d in gap_nc if "rel_gap" in d]
        lbs = [d["LB"] for d in gap_nc]
        ubs = [d["UB"] for d in gap_nc]
        iters = [d["n_iters"] for d in gap_nc if "n_iters" in d]

        print(f"\n--- gap_not_converged details ---")
        print(f"  Count: {len(gap_nc)}")
        if rel_gaps:
            arr = np.array(rel_gaps)
            print(f"  rel_gap: min={arr.min():.4e}, mean={arr.mean():.4e}, max={arr.max():.4e}")
        if lbs:
            arr = np.array(lbs)
            print(f"  LB: min={arr.min():.4f}, mean={arr.mean():.4f}, max={arr.max():.4f}")
        if ubs:
            arr = np.array(ubs)
            print(f"  UB: min={arr.min():.4f}, mean={arr.mean():.4f}, max={arr.max():.4f}")
        if iters:
            arr = np.array(iters)
            print(f"  iterations: min={arr.min()}, mean={arr.mean():.1f}, max={arr.max()}")

        i_counts = Counter(d["i"] for d in gap_nc)
        print(f"  By iteration i:")
        for i in sorted(i_counts.keys()):
            print(f"    i={i}: {i_counts[i]} cases")

        t_counts = Counter(d["t"] for d in gap_nc)
        print(f"  By stage t:")
        for t in sorted(t_counts.keys()):
            print(f"    t={t}: {t_counts[t]} cases")

    # LB/pi0 ratio too small
    lb_small = [d for d in not_converged_details if d["reason"] == "LB_pi0_ratio_too_small"]
    if lb_small:
        print(f"\n--- LB/pi0 ratio too small details ---")
        print(f"  Count: {len(lb_small)}")
        lbs = np.array([d["LB"] for d in lb_small])
        pi0s = np.array([d["pi0_star"] for d in lb_small if d["pi0_star"] is not None])
        print(f"  LB: min={lbs.min():.6f}, mean={lbs.mean():.6f}, max={lbs.max():.6f}")
        if len(pi0s) > 0:
            print(f"  pi0: min={pi0s.min():.6f}, mean={pi0s.mean():.6f}, max={pi0s.max():.6f}")
            ratios = lbs[:len(pi0s)] / pi0s
            print(f"  LB/pi0: min={ratios.min():.6f}, mean={ratios.mean():.6f}, max={ratios.max():.6f}")

        i_counts = Counter(d["i"] for d in lb_small)
        print(f"  By iteration i:")
        for i in sorted(i_counts.keys()):
            print(f"    i={i}: {i_counts[i]} cases")

    # pi0 too small
    pi0_small = [d for d in not_converged_details if d["reason"] == "pi0_too_small"]
    if pi0_small:
        print(f"\n--- pi0 too small details ---")
        print(f"  Count: {len(pi0_small)}")
        pi0s = [d["pi0_star"] for d in pi0_small if d["pi0_star"] is not None]
        if pi0s:
            arr = np.array(pi0s)
            print(f"  pi0: min={arr.min():.6e}, mean={arr.mean():.6e}, max={arr.max():.6e}")

        i_counts = Counter(d["i"] for d in pi0_small)
        print(f"  By iteration i:")
        for i in sorted(i_counts.keys()):
            print(f"    i={i}: {i_counts[i]} cases")


if __name__ == "__main__":
    main()
