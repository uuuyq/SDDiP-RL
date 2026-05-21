from bundle_RL.script.default_feature.bundle_env import BundleDualEnv
from bundle_RL.script.lag_problem import MasterProblem


def create_env(logger, config, tolerance=1e-5, verbose=False):
    """创建单个环境（使用 config 中的 n 参数）"""

    env = BundleDualEnv(
        logger=logger,
        config=config,
        n=config.n,  # 直接使用 config 中的 realization 索引
        state_dim=config.N_VARS,
        K=20,
        verbose=verbose  # 是否输出详细日志
    )
    master = MasterProblem(logger, config.N_VARS, tolerance=tolerance)
    return env, master