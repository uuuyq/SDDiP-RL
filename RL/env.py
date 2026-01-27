import numpy as np
from gymnasium import spaces
from sddip.sddip.parameters import Parameters
import gymnasium as gym

class SCUCEnv(gym.Env):
    def __init__(self, parameters: Parameters):
        super().__init__()

        self.n_gen = parameters.n_gens
        self.n_soc = parameters.n_storages
        self.backsight_periods = parameters.backsight_periods
        self.max_y = parameters.pg_max  # 每台发电机最大功率
        self.min_y = parameters.pg_min  # 每台发电机最小功率
        # Storage charge/discharge rate limits
        self.rc_max = parameters.rc_max
        self.rdc_max = parameters.rdc_max
        # Maximum state of charge
        self.soc_max = parameters.soc_max
        # Charge/discharge efficiencies
        self.eff_c = parameters.eff_c
        self.eff_dc = parameters.eff_dc
        self.xi_dim = parameters.n_buses * 2  # 随机变量维度 ξ_t  ：Pd，Re
        # scenarios
        self.p_d = parameters.p_d
        self.re = parameters.re
        # cost coefficient
        self.coefficient = parameters.cost_coeffs
        self.min_up_times = parameters.min_up_time
        self.min_down_times = parameters.min_down_time
        self.n_lines = parameters.n_lines
        self.n_buses = parameters.n_buses
        self.ptdf = parameters.ptdf
        self.pl_max = parameters.pl_max
        self.generators_at_bus = parameters.gens_at_bus
        self.storages_at_bus = parameters.storages_at_bus
        self.max_rate_up = parameters.r_up
        self.max_rate_down = parameters.r_down
        self.startup_rate = parameters.r_su
        self.shutdown_rate = parameters.r_sd


        # ---------- 状态空间 ----------  状态空间不需要精确约束每个物理量
        # state = soc + x_bs + x + y + ξ_t + t
        state_dim = self.n_soc + sum(self.backsight_periods) + self.n_gen * 2 + self.xi_dim + 1
        self.observation_space = spaces.Box(
            low=0,
            high=1e5,
            shape=(state_dim,),
            dtype=np.float32
        )

        # ---------- 动作空间 ----------
        # 动作：x[g] (binary) + y[g] (continuous)  + 充放电功率
        action_low = np.concatenate([
            np.zeros(self.n_gen),  # x[g]
            np.zeros(self.n_gen),  # y[g]
            -self.rdc_max * np.ones(self.n_soc),  # 储能净充放电
        ])
        action_high = np.concatenate([
            np.ones(self.n_gen),  # x[g]
            self.max_y * np.ones(self.n_gen),
            self.rc_max * np.ones(self.n_soc),
        ])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)


    def reset(self, seed=None, options=None):
        # 初始化状态  x y x_bs 初始化为0，soc初始化为0.5max
        super().reset(seed=seed)
        # 初始化状态
        self.soc = 0.5 * np.array(self.soc_max)  # 初始化为 50% SOC
        self.x = np.zeros(self.n_gen)
        self.y = np.zeros(self.n_gen)
        self.x_bs = [np.zeros(k) for k in self.backsight_periods]  # 历史 back-sight 决策
        self.t = 0
        # xi_init
        self.xi = self.sample_xi(self.t + 1)  # t=1
        return self._get_state(), {}

    def _get_state(self):
        state = np.concatenate([
            self.soc,
            np.concatenate(self.x_bs) if self.x_bs else np.array([]),
            self.x,
            self.y,
            self.xi
        ])
        return state

    def sample_xi(self, t):
        # 随机选择当前阶段的一个场景索引
        idx = np.random.randint(len(self.p_d[t]))
        # 拼接 Pd + Re
        xi = np.concatenate([self.p_d[t][idx], self.re[t][idx]])
        return xi

    def step(self, action):
        # ---------------- 动作解析 ----------------
        idx = 0
        x = (action[idx:idx + self.n_gen] > 0.5).astype(float)  # 开机二值化
        idx += self.n_gen
        y = action[idx:idx + self.n_gen]  # 发电功率
        idx += self.n_gen
        u_s = action[idx:idx + self.n_soc]  # 储能净充放电功率


        # ---------------- SOC 更新 ----------------
        soc_prev = self.soc
        for s in range(self.n_soc):
            if u_s[s] >= 0:
                self.soc[s] += self.eff_c[s] * u_s[s]  # 充电
            else:
                self.soc[s] += u_s[s] / self.eff_dc[s]  # 放电

        # ---------------- x_bs 滚动更新 ----------------
        x_bs_prev = self.x_bs
        if self.x_bs:
            for i in range(len(self.x_bs)):
                self.x_bs[i] = np.roll(self.x_bs[i], -1)  # 左移
                self.x_bs[i][-1] = x[i % self.n_gen]

        reward = self.reward((self.x, self.y, x_bs_prev, soc_prev, x, y, self.x_bs, self.soc, u_s, self.xi))


        # ---------------- 更新状态 ----------------
        self.x = x
        self.y = y
        self.t += 1
        self.xi = self.sample_xi(self.t + 1)

        # ---------------- 终止条件 ----------------
        max_steps = 24  # 24个阶段
        terminated = self.t >= max_steps  # t 从0开始计数，第24个阶段的解不需要
        truncated = False  # 如果没有额外的截断条件

        next_state = self._get_state()
        return next_state, reward, terminated, truncated, {}


    def reward(self, args):
        x_prev = args[0]
        y_prev = args[1]
        x_bs_prev = args[2]
        soc_prev = args[3]
        x = args[4]
        y = args[5]
        x_bs = args[6]
        soc = args[7]
        u_s = args[8]
        xi = args[9]
        p_d = xi[:len(xi) / 2]
        re = xi[len(xi) / 2 :]

        s_up = (1 - x_prev) * x  # 启动
        s_down = x_prev * (1 - x)  # 停机

        # ---------------- 1 发电功率惩罚 ----------------
        y_violation = np.sum(np.maximum(0, y - x * self.max_y) + np.maximum(0, x * self.min_y - y))
        # ---------------- 2 SOC容量惩罚 ----------------
        soc_violation = np.sum(np.maximum(0, self.soc - self.soc_max) + np.maximum(0, 0 - self.soc))

        # ---------------- 3 功率平衡约束 ----------------
        total_generation = np.sum(y)
        total_discharge = np.sum([abs(u_s[s]) for s in range(self.n_soc) if u_s[s] < 0])
        total_charge = np.sum([u_s[s] for s in range(self.n_soc) if u_s[s] > 0])
        total_renewable = np.sum(re)  # 可再生发电
        total_demand = np.sum(p_d)  # 负荷
        balance_violation = total_generation + total_discharge - total_charge + total_renewable - total_demand

        # ---------------- 4 最小启停时间约束 ----------------
        x_bs_violation = self.x_bs_violation(x_bs, self.min_up_times, self.min_down_times)
        # ---------------- 5 power flow 约束 ----------------
        flow_violation = self.power_flow_violation(u_s, p_d, re)
        # ---------------- 6 ramping 约束 ----------------
        ramp_violation = self.ramp_violation(x_prev, x, y_prev, y)


        r = 0
        r += np.concatenate(y, s_up, s_down) * self.coefficient[:3 * self.n_gen]  # 发电成本 + 启动停机成本
        penalty = [self.coefficient[-1] * 6]
        r += np.array([x_bs_violation, flow_violation, ramp_violation, balance_violation, soc_violation,
                            y_violation, soc_violation]) * penalty  # 约束惩罚项


        return -r


    def x_bs_violation(self, x_bs, min_up_times, min_down_times):
        """
        x_bs: list of arrays，每个机组历史back-sight状态，0/1
        min_up_times / min_down_times: 每台机组最小开停机时间
        """
        violations = []

        for g in range(len(x_bs)):
            seq = x_bs[g]
            up_violation = 0
            down_violation = 0

            # 1. 找所有潜在的关机点，检查前面连续开机长度
            for t in range(len(seq)):
                if seq[t] == 0:  # 关机
                    # 往前找连续开机长度
                    count = 0
                    for k in range(t, -1, -1):
                        if seq[k] == 1:
                            count += 1
                        else:
                            break
                    if count < min_up_times[g]:
                        up_violation += min_up_times[g] - count

            # 2. 找所有潜在的启动点，检查前面连续停机长度
            for t in range(len(seq)):
                if seq[t] == 1:  # 启动
                    # 往前找连续停机长度
                    count = 0
                    for k in range(t, -1, -1):
                        if seq[k] == 0:
                            count += 1
                        else:
                            break
                    if count < min_down_times[g]:
                        down_violation += min_down_times[g] - count

            violations.append((up_violation, down_violation))

        return violations

    def power_flow_violation(self, u_s, p_d, re):
        # ---------- 计算每条线路的功率流 ----------
        line_flows = np.zeros(self.n_lines)

        for l in range(self.n_lines):
            flow = 0
            for b in range(self.n_buses):
                # 发电机贡献
                gen_contrib = np.sum([self.y[g] for g in self.generators_at_bus[b]])
                # 储能贡献（放电为正，充电为负）
                storage_contrib = 0
                for s in self.storages_at_bus[b]:
                    if u_s[s] >= 0:
                        # 充电，向线路注入功率为负（吸收能量）
                        storage_contrib -= u_s[s]  # 或乘充电效率 eff_c[s]，视建模而定
                    else:
                        # 放电，向线路注入功率为正
                        storage_contrib += -u_s[s] * self.eff_dc[s]  # u_s[s] <0，乘放电效率
                # 净注入功率 = 发电 + 储能 - 负荷 + 可再生
                net_injection = gen_contrib + storage_contrib - p_d[b] + re[b]
                # PTDF 乘法得到线路 l 的流量
                flow += self.ptdf[l, b] * net_injection
            line_flows[l] = flow

        # ---------- 功率流违规量（惩罚） ----------
        flow_violation = np.sum(np.maximum(0, line_flows - self.pl_max) +
                                np.maximum(0, -self.pl_max - line_flows))

        return flow_violation


    def ramp_violation(self, x_prev, y_prev, x, y):
        # ---------------- 计算爬坡率违规 ----------------
        ramp_violation = 0

        for g in range(self.n_gen):
            # 上升爬坡
            max_up = self.max_rate_up[g]
            if x[g] == 1:  # 仅当机组开机时才考虑爬坡
                ramp_up = y[g] - y_prev[g]
                if ramp_up > max_up:
                    ramp_violation += ramp_up - max_up

            # 下降爬坡
            max_down = self.max_rate_down[g]
            if x_prev[g] == 1:  # 上阶段开机
                ramp_down = y_prev[g] - y[g]
                if ramp_down > max_down:
                    ramp_violation += ramp_down - max_down

            # 启动爬坡
            if x[g] == 1 and x_prev[g] == 0:  # 启动
                ramp_violation += max(0, y[g] - self.startup_rate[g])

            # 停机爬坡
            if x[g] == 0 and x_prev[g] == 1:  # 停机
                ramp_violation += max(0, self.shutdown_rate[g] - y_prev[g])

        return ramp_violation