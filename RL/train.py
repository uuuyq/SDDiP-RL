from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from env import SCUCEnv
from sddip.sddip.parameters import Parameters
def train():

    path = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t06_n06")
    parameters = Parameters(path)

    env = SCUCEnv(parameters)

    def make_env(seed=None):
        def _init():
            env = SCUCEnv(parameters)  # 每次都 new 一个
            if seed is not None:
                env.seed(seed)
            return env

        return _init

    from stable_baselines3.common.env_util import make_vec_env

    vec_env = make_vec_env(make_env(), n_envs=4)

    model = PPO(
        "MlpPolicy",  # 多层感知机策略
        vec_env,  # 环境
        verbose=1,  # 输出训练信息
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=10,
    )

    model.learn(total_timesteps=100000)  # 总训练步数

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # 可以记录每阶段动作、功率、SOC等


if __name__ == "__main__":
    train()