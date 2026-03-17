import numpy as np
import cvxpy as cp

class OptimizationSolvers:
    '''
    包含论文中的三大核心优化子函数：
    1. MPMM for F (波束跳动模式)
    2. SCA for P (功率分配与频段控制)
    3. QP for B (星间链路负载均衡)
    '''
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.S = config.NUM_SATELLITES
        self.K = config.NUM_CELLS
        self.L = config.NUM_FREQUENCY_SEGMENTS
        
    def _calculate_interference(self, P_matrix, F_pattern, h_matrix, s, k, l):
        interference = 0.0
        # 根据论文式(1)，小区k受到的干扰仅来自能覆盖小区k的卫星集合(PHI_K)
        for s_idx in self.config.PHI_K[k]:
            # 干扰可能来自该卫星发往其他小区的波束
            for k_idx in range(self.K):
                if not (s_idx == s and k_idx == k):
                    interference += abs(h_matrix[s_idx, k, k_idx])**2 * P_matrix[l, s_idx, k_idx] * F_pattern[s_idx, k_idx]
        return interference

    def solve_F_MPMM(self, F_prev, P_fixed, B_fixed, h_matrix, g_matrix, Q_lengths):
        """ Algorithm 2: MPMM for F Matrix """
        F_var = cp.Variable((self.S, self.K), nonneg=True)
        alpha = np.ones((self.S, self.K)) * (0.1 / self.config.L_0)                     # 拉格朗日乘子系数
        beta = 0.5 / self.config.L_0                                                  # 惩罚函数系数
        rho = 1.1                                                   # 惩罚函数更新速率
        Theta_rounds = getattr(self.config, 'MPMM_THETA_ROUNDS', 5)
        
        # 预先计算此时隙负载均衡调整后的临时队列限制 Q_temp (考虑到此时B矩阵被固定，计算先验排队长度d_sk)
        d_sk = np.zeros((self.S, self.K))
        for s_idx in range(self.S):
            for k_idx in range(self.K):
                d_sk[s_idx, k_idx] = sum([B_fixed[r, s_idx, k_idx] for r in self.config.PHI_K[k_idx]])
        Q_temp = np.maximum(0.0, Q_lengths + d_sk)
        
        # 覆盖范围掩码 valid_mask，确保 F_var 只能在合法的卫星-小区对上取值
        valid_mask = np.zeros((self.S, self.K))
        for k in range(self.K):
            for s in self.config.PHI_K[k]:
                valid_mask[s, k] = 1.0

        W_band = self.config.BANDWIDTH_PER_SEGMENT
        noise = self.env.channel_model.noise_power * W_band
        # 缩放因子，R_sk(bps) * time_scale，把传输速率转化为吞吐的包数
        time_scale = self.config.TIME_SLOT_DURATION / self.config.PACKET_SIZE

        # 引入虚拟/参考功率进行探测，避免上一次P为0导致死锁。假设可用功率被平均分配。
        P_virtual = np.ones_like(P_fixed) * (self.config.MAX_POWER_PER_SAT / (self.config.NUM_BEAMS_PER_SAT * self.L))

        # 信号接收矩阵，在确定P的前提下，计算对于每个小区k在频段l上，收到卫星s_idx到小区k_idx波束的信号大小（k_idx=k时为有用信号）
        HP = np.zeros((self.K, self.S, self.K, self.L))
        for k in range(self.K):
            for s_idx in self.config.PHI_K[k]:
                for k_idx in range(self.K):
                    for l in range(self.L):
                        HP[k, s_idx, k_idx, l] = abs(h_matrix[s_idx, k, k_idx])**2 * P_virtual[l, s_idx, k_idx]

        # 有用信号矩阵，在确定P的前提下，计算对于每个小区k在频段l上，来自卫星s的有用信号强度
        H_self_P = np.zeros((self.S, self.K, self.L))
        for s in range(self.S):
            for k in range(self.K):
                for l in range(self.L):
                    H_self_P[s, k, l] = abs(h_matrix[s, k, k])**2 * P_virtual[l, s, k]

        Z_w_limit = 10 ** (self.config.Z_MAX_DBW / 10.0)
        K_G = getattr(self.config, 'K_G', [])
        L_K = getattr(self.config, 'L_K', {})
        # 各个GSO小区受到卫星s到小区k波束的干扰矩阵
        GSO_coeffs = []
        for k_g in K_G:
            overlap_bands = L_K.get(k_g, range(self.L))
            coeff = np.zeros((self.S, self.K))
            for l in overlap_bands:
                for s in range(self.S):
                    for k in range(self.K):
                        coeff[s, k] += (abs(g_matrix[s, k_g, k])**2) * P_virtual[l, s, k]
            GSO_coeffs.append(coeff)
            
        F_best = F_prev.copy()
        for _ in range(Theta_rounds):
            # 约束13c
            constraints = [F_var <= valid_mask]  
            for s in range(self.S):
                # 约束13d
                constraints += [cp.sum(F_var[s, :]) <= self.config.NUM_BEAMS_PER_SAT]   
            for coeff in GSO_coeffs:
                # 约束13g
                constraints += [cp.sum(cp.multiply(coeff, F_var)) <= Z_w_limit] 

            f_surrogate = cp.multiply(1 - 2 * F_best, F_var) + F_best**2
            J_mp = cp.sum(cp.multiply(alpha, f_surrogate)) + beta * cp.sum(cp.square(f_surrogate))
            
            # 引入真实传输变量 X_var (体现原论文式31中的辅助变量转换，以及式32a中对每个链路的约束)
            X_var = cp.Variable((self.S, self.K))
            # X_var = cp.Variable((self.S, self.K), nonneg=True)
            # slack_var = cp.Variable((self.S, self.K), nonneg=True)
            constraints += [X_var <= Q_temp] 

            # 式1的I_skl计算
            I_fixed = np.zeros((self.S, self.K, self.L))
            for k in range(self.K):
                # 直接计算小区k受到PHI_K内所有卫星的信号总和，包括了干扰和有用信号，再减去有用信号得到总干扰
                total_I_k = np.zeros(self.L)
                for s_idx in self.config.PHI_K[k]:
                    for k_idx in range(self.K):
                        total_I_k += HP[k, s_idx, k_idx, :] * F_best[s_idx, k_idx]
                for s in self.config.PHI_K[k]:
                    I_fixed[s, k, :] = np.maximum(0, total_I_k - HP[k, s, k, :] * F_best[s, k])

            # 式30a、30b，矩阵形式计算，计算每个小区k在频段l上，来自卫星s的相关参数
            gamma_best = H_self_P / (noise + I_fixed + 1e-12)
            phi_best = gamma_best / (1 + gamma_best)
            zeta_best = np.log2(1 + gamma_best) - phi_best * np.log2(gamma_best + 1e-12)

            log2_e = np.log2(np.e)
            term1 = phi_best * np.log2(np.e * gamma_best + 1e-12) + zeta_best
            term2_coeff = log2_e * phi_best * gamma_best / (H_self_P + 1e-12)
            
            # 计算式31中除了I_var（包括F_var的I）其他项
            Term1_sum = np.sum(W_band * term1, axis=2)                   
            Noise_sum = np.sum(W_band * term2_coeff * noise, axis=2)     

            for s in range(self.S):
                for k in range(self.K):
                    if valid_mask[s, k] == 0:
                        # 如果这个卫星-小区对不合法，直接约束X_var为0，并跳过后续计算
                        constraints += [X_var[s, k] == 0]
                        continue
                    # 结合式1计算式31中I_var，并得到surrogate_R_sk
                    W_term = W_band * term2_coeff[s, k, :]               
                    weight_sk = np.sum(HP[k, :, :, :] * W_term, axis=-1) 
                    weight_sk[s, k] = 0.0 
                    surrogate_R_sk = Term1_sum[s, k] * F_var[s, k] - Noise_sum[s, k] - cp.sum(cp.multiply(weight_sk, F_var))
                    rate_sk_bound = surrogate_R_sk * time_scale
                    constraints += [X_var[s, k] <= rate_sk_bound]
                    # constraints += [X_var[s, k] <= rate_sk_bound + slack_var[s, k]]
            
            # utility_surrogate 表示论文最小化目标式19中q_sk*x_sk这一项，仅表示这一项是因为固定P、B时其他项为定值，最小化目标中仅剩余此项需要优化
            utility_surrogate = cp.sum(cp.multiply(Q_lengths, X_var))  
            
            # J_mp 即对应式(29)完整展开项，内部已包含 alpha 和 beta 相关惩罚
            objective = cp.Minimize(-utility_surrogate / self.config.L_0 + J_mp)
            # objective = cp.Minimize(-utility_surrogate / self.config.L_0 + J_mp + 1e5 * cp.sum(slack_var))
            
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.SCS, warm_start=True)
                if F_var.value is not None:
                    F_best = F_var.value
                    print(f"        [Theta {_}] Current F_best sum: {np.sum(F_best):.2f}, Max Val: {np.max(F_best):.2f}")
                    alpha += 2 * beta * F_best * (1 - F_best)
                    beta *= rho
                else:
                    break
            except Exception:
                break

        F_quantized = np.zeros((self.S, self.K))
        for s in range(self.S):
            top_k_indices = np.argsort(F_best[s, :])[-self.config.NUM_BEAMS_PER_SAT:]
            for k_idx in top_k_indices:
                F_quantized[s, k_idx] = 1.0
                    
        return F_quantized

    def solve_P_SCA(self, F_fixed, P_prev, B_fixed, h_matrix, g_matrix, Q_lengths):
        ''' Algorithm 3: SCA for Power '''
        # 约束13e
        P_var = cp.Variable((self.L, self.S, self.K), nonneg=True)
        constraints = []
        for s in range(self.S):
            # 约束13f
            constraints += [cp.sum(P_var[:, s, :]) <= self.config.MAX_POWER_PER_SAT]
            
        Z_w_limit = 10 ** (self.config.Z_MAX_DBW / 10.0)
        K_G = getattr(self.config, 'K_G', [])
        L_K = getattr(self.config, 'L_K', {})
        for k_g in K_G:
            overlap_bands = L_K.get(k_g, range(self.L))
            interference_to_gso = 0
                
            for l in overlap_bands:
                for s in range(self.S):
                    for k in range(self.K):
                        # g_matrix 维度为 [S, K, K], k_g为受干扰GSO所在小区，k为发射目标小区
                        Z_gk = (abs(g_matrix[s, k_g, k])**2) * P_var[l, s, k] * F_fixed[s, k]
                        interference_to_gso += Z_gk
                # 约束13g，但存在问题
                constraints += [interference_to_gso <= Z_w_limit]

        expected_rates = np.zeros((self.S, self.K))
        for s in range(self.S):
            for k in range(self.K):
                if F_fixed[s, k] > 0:
                    for l in range(self.L):
                        I_skl = self._calculate_interference(P_prev, F_fixed, h_matrix, s, k, l)
                        noise = self.env.channel_model.noise_power * self.config.BANDWIDTH_PER_SEGMENT
                        gain_factor = abs(h_matrix[s, k, k])**2 / (noise + I_skl + 1e-15)
                        expected_rates[s, k] += gain_factor * self.config.BANDWIDTH_PER_SEGMENT

        rate_expr = 0
        energy_expr = 0
        for s in range(self.S):
            for k in range(self.K):
                for l in range(self.L):
                    coeff = expected_rates[s,k] * (self.config.TIME_SLOT_DURATION / self.config.PACKET_SIZE)
                    rate_expr += Q_lengths[s, k] * (coeff * P_var[l, s, k]) * F_fixed[s, k]
                    energy_expr += P_var[l, s, k] * self.config.TIME_SLOT_DURATION

        # rate_expr 表示论文最小化目标式19第一项中q_sk*x_sk这一项，仅表示这一项是因为固定F、B时其他项为定值，最小化目标中仅剩余此项需要优化
        # energy_expr 表示论文最小化目标式19第二项中卫星与地面波束能量消耗项，仅表示这一项是因为固定F、B时星间链路耗能为定值
        objective = cp.Minimize(-rate_expr / self.config.L_0 + (self.config.V / self.config.E_0) * energy_expr)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS)
        except Exception:
            pass 

        P_new = P_prev.copy()
        if P_var.value is not None:
             P_new = np.clip(P_var.value, 0, None)
             
        return P_new

    def solve_B_QP(self, F_fixed, P_fixed, B_prev, Q_lengths):
        ''' QP for Load Balancing Matrix '''
        B_var = cp.Variable((self.S, self.S, self.K))
        
        constraints = []
        for k in range(self.K):
            for r in range(self.S):
                for s in range(self.S):
                    if r == s:
                        constraints += [B_var[r, s, k] == 0]
                    else:
                        constraints += [B_var[r, s, k] == -B_var[s, r, k]]
                        
        for r in range(self.S):
            for s in range(self.S):
                if r < s:
                    sum_abs = cp.sum(cp.abs(B_var[r, s, :]))
                    constraints += [sum_abs <= getattr(self.config, 'ISL_MAX_TRANSFER_PKTS', 10)]
        
        obj_expr = 0
        energy_isl_expr = 0
        for s in range(self.S):
            for k in range(self.K):
                d_sk = cp.sum(B_var[:, s, k]) 
                obj_expr += (Q_lengths[s, k] * d_sk + 0.5 * cp.square(d_sk)) / self.config.L_0

        for r in range(self.S):
            for s in range(self.S):
                if r < s:
                     c_rs = cp.sum(cp.abs(B_var[r, s, :]))
                     t_rs = c_rs * self.config.PACKET_SIZE / getattr(self.config, 'ISL_DATA_RATE', 1e9)
                     energy_isl_expr += getattr(self.config, 'ISL_POWER_CONSUMPTION', 5.0) * t_rs

        objective = cp.Minimize(obj_expr + (self.config.V / self.config.E_0) * energy_isl_expr)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP) 
        except Exception:
            try:
                prob.solve(solver=cp.SCS)
            except Exception:
                pass

        B_res = B_prev.copy()
        if B_var.value is not None:
             B_res = B_var.value 
        else:
             B_res = np.zeros((self.S, self.S, self.K))
             
        return B_res
