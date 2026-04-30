"""
config.py

公共配置文件，包含 trial point 和问题参数
"""

from pathlib import Path

from sddip.sddip import parameters

# 问题参数
T = 5
N_VARS = 13

# Trial point
X_TRIAL = [-0.0, 1.0, 1.0]
Y_TRIAL = [0.0, 131.60809087723158, 45.0]
X_BS_TRIAL = [[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
SOC_TRIAL = [0.0]

# 数据路径
PATH = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t06_n06")

# 初始化 problem_params
PROBLEM_PARAMS = parameters.Parameters(PATH)


# t = 1
# x_trial = [1.0, 1.0, 1.0]
# y_trial = [71.52627531002818, 59.02627531002818, 66.52627531002818]
# x_bs_trial = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
# soc_trial = [5.0]