from bundle_RL.script.default_feature.bundle_env import BundleDualEnv
from bundle_RL.script.lag_problem import MasterProblem


def create_env(logger, config, tolerance=1e-5, verbose=False, K=20):
    """创建单个环境（使用 config 中的 n 参数）"""

    env = BundleDualEnv(
        logger=logger,
        config=config,
        n=config.n,
        state_dim=config.N_VARS,
        K=K,
        verbose=verbose
    )
    master = MasterProblem(logger, config.N_VARS, tolerance=tolerance)
    return env, master