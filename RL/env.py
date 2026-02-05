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
        self.pl_max = np.array(parameters.pl_max)
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
            self.min_y,  # y[g]
            -np.array(self.rdc_max),  # 储能净充放电
        ])
        action_high = np.concatenate([
            np.ones(self.n_gen),  # x[g]
            self.max_y,
            self.rc_max,
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
            self.xi,
            np.array([self.t], dtype=np.float32)
        ])
        return state

    def sample_xi(self, t):
        t = t - 1  # 参数t代表阶段，对应list中的t-1
        # 随机选择当前阶段的一个场景索引
        idx = np.random.randint(len(self.p_d[t]))
        # 拼接 Pd + Re
        xi = np.concatenate([self.p_d[t][idx], self.re[t][idx]])
        return xi

    def step(self, action):
        # ---------------- 动作解析 ----------------
        idx = 0
        # x = (action[idx:idx + self.n_gen] > 0.5).astype(float)  # 开机二值化
        x = action[idx:idx + self.n_gen]  # 开机二值化
        idx += self.n_gen
        y = action[idx:idx + self.n_gen]  # 发电功率
        idx += self.n_gen
        u_s = action[idx:idx + self.n_soc]  # 储能净充放电功率

        # ---------------- SOC 更新 ----------------
        soc_prev = self.soc
        for s in range(self.n_soc):
            if u_s[s] >= 0:
                self.soc[s] += self.eff_c[s] * u_s[s]  # 充电  delta T = 1
            else:
                self.soc[s] += u_s[s] / self.eff_dc[s]  # 放电

        # ---------------- x_bs 滚动更新 ----------------
        x_bs_prev = [arr.copy() for arr in self.x_bs]
        if self.x_bs:
            for g in range(self.n_gen):
                self.x_bs[g] = np.roll(self.x_bs[g], -1)
                self.x_bs[g][-1] = x[g]

        reward = self.reward((self.x, self.y, x_bs_prev, soc_prev, x, y, self.x_bs, self.soc, u_s, self.xi))

        # ---------------- 更新状态 ----------------
        self.x = x
        self.y = y
        self.t += 1
        max_steps = 24  # 24个阶段
        if self.t < max_steps:
            self.xi = self.sample_xi(self.t + 1)

        # ---------------- 终止条件 ----------------
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

        p_d = xi[:xi.shape[0] // 2]
        re = xi[xi.shape[0] // 2 :]

        s_up = (1 - x_prev) * x  # 启动
        s_down = x_prev * (1 - x)  # 停机

        binary_violation = np.sum(x * (1 - x))

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
        balance_violation = abs(total_generation + total_discharge - total_charge + total_renewable - total_demand)

        # ---------------- 4 最小启停时间约束 ----------------
        x_bs_violation = self.x_bs_violation(x_bs_prev, x_prev, x, self.min_up_times, self.min_down_times)
        # ---------------- 5 power flow 约束 ----------------
        flow_violation = self.power_flow_violation(u_s, p_d, re)
        # ---------------- 6 ramping 约束 ----------------
        ramp_violation = self.ramp_violation(x_prev, y_prev, x, y)


        r = 0
        r += np.sum(np.concatenate([y, s_up, s_down]) * self.coefficient[:3 * self.n_gen])  # 发电成本 + 启动停机成本
        penalty = self.coefficient[-1]
        # penalty =
        r += np.sum(binary_violation) * penalty
        r += y_violation * penalty
        r += soc_violation * penalty
        # r += balance_violation * penalty
        # r += np.sum(x_bs_violation) * penalty
        # r += np.sum(flow_violation) * penalty
        # r += np.sum(ramp_violation) * penalty

        return -r


    def x_bs_violation(self, x_bs_prev, x_prev, x, min_up, min_down):
        violation = np.zeros(self.n_gen, dtype=np.float32)

        for g in range(self.n_gen):
            hist = x_bs_prev[g]

            # ---------- 启动 ----------
            if x_prev[g] == 0 and x[g] == 1:
                # 过去连续停机时间
                off_time = 0
                for v in reversed(hist):
                    if v == 0:
                        off_time += 1
                    else:
                        break
                violation[g] = max(0, min_down[g] - off_time)

            # ---------- 停机 ----------
            elif x_prev[g] == 1 and x[g] == 0:
                # 过去连续开机时间
                on_time = 0
                for v in reversed(hist):
                    if v == 1:
                        on_time += 1
                    else:
                        break
                violation[g] = max(0, min_up[g] - on_time)

        return violation

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
        flow_violation = (
            np.maximum(0.0, line_flows - self.pl_max) +
            np.maximum(0.0, -self.pl_max - line_flows)
        )

        return flow_violation

    def ramp_violation(self, x_prev, y_prev, x, y):
        violation = np.zeros(self.n_gen, dtype=np.float32)

        for g in range(self.n_gen):
            # ---------- 上升爬坡 ----------
            if x[g] == 1 and x_prev[g] == 1:
                ramp_up = y[g] - y_prev[g]
                if ramp_up > self.max_rate_up[g]:
                    violation[g] += ramp_up - self.max_rate_up[g]
            # ---------- 下降爬坡 ----------
            if x[g] == 1 and x_prev[g] == 1:
                ramp_down = y_prev[g] - y[g]
                if ramp_down > self.max_rate_down[g]:
                    violation[g] += ramp_down - self.max_rate_down[g]
            # ---------- 启动爬坡 ----------
            if x_prev[g] == 0 and x[g] == 1:
                violation[g] += max(0.0, y[g] - self.startup_rate[g])
            # ---------- 停机爬坡 ----------
            if x_prev[g] == 1 and x[g] == 0:
                violation[g] += max(0.0, y_prev[g] - self.shutdown_rate[g])

        return violation
