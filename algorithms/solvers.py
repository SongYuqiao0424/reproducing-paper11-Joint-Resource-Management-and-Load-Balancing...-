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
        ''' Algorithm 2: MPMM for F Matrix '''
        F_var = cp.Variable((self.S, self.K), nonneg=True)
        alpha = np.ones((self.S, self.K)) * 0.1                     # 拉格朗日乘子系数
        beta = 0.5                                                  # 惩罚函数系数
        rho = 1.1                                                   # 惩罚函数更新速率
        Theta_rounds = getattr(self.config, 'MPMM_THETA_ROUNDS', 5)
        
        # 预先计算此时隙负载均衡调整后的临时队列限制 Q_temp (考虑到此时B矩阵被固定，计算先验排队长度d_sk)
        d_sk = np.zeros((self.S, self.K))
        for s_idx in range(self.S):
            for k_idx in range(self.K):
                d_sk[s_idx, k_idx] = sum([B_fixed[r, s_idx, k_idx] for r in self.config.PHI_K[k_idx]])
        Q_temp = np.maximum(0.0, Q_lengths + d_sk)
        
        F_best = F_prev.copy()
        for _ in range(Theta_rounds):
            # 约束25
            constraints = [F_var <= 1.0]
            for s in range(self.S):
                # 约束13d
                constraints += [cp.sum(F_var[s, :]) <= self.config.NUM_BEAMS_PER_SAT]   
                    
            # 约束13c：若卫星 s 无法覆盖小区 k，则强制 f_{s,k} == 0
            for k in range(self.K):
                for s in range(self.S):
                    if s not in self.config.PHI_K[k]:
                        constraints += [F_var[s, k] == 0]
            

            # 约束13g：GSO系统的干扰限制 (对F进行优化时)
            Z_w_limit = 10 ** (self.config.Z_MAX_DBW / 10.0)
            K_G = getattr(self.config, 'K_G', [])
            L_K = getattr(self.config, 'L_K', {})
            for k_g in K_G:
                overlap_bands = L_K.get(k_g, range(self.L))
                interference_to_gso = 0
                for l in overlap_bands:
                    for s in range(self.S):
                        for k in range(self.K):
                            interference_to_gso += (abs(g_matrix[s, k_g, k])**2) * P_fixed[l, s, k] * F_var[s, k]
                constraints += [interference_to_gso <= Z_w_limit]
                
        # 根据论文式(29)完整构造MM替代惩罚函数 J_mp，为原函数上界，在F上是凸函数
            # 内部替代项: (1 - 2 * F_best) * F_var + F_best^2
            f_surrogate = cp.multiply(1 - 2 * F_best, F_var) + F_best**2
            J_mp = cp.sum(cp.multiply(alpha, f_surrogate)) + beta * cp.sum(cp.square(f_surrogate))
            
            # 引入真实传输变量 X_var (体现原论文式31中的辅助变量转换，以及式32a中对每个链路的约束)
            X_var = cp.Variable((self.S, self.K), nonneg=True)
            # 约束13l
            constraints += [X_var <= Q_temp]                                               # 队列存储上限约束 (对应约束13j)

            # 按照论文式(30)~(31)，这里需要为每个 (s, k) 链路构造替代函数及约束
            for s in range(self.S):
                for k in range(self.K):
                    W = self.config.BANDWIDTH_PER_SEGMENT
                    # surrogate_R_sk表示构造的R_sk替代函数，为原函数下界，在F上是凹函数
                    surrogate_R_sk = 0
                    for l in range(self.L):
                        I_skl = self._calculate_interference(P_fixed, F_best, h_matrix, s, k, l)
                        noise = self.env.channel_model.noise_power * W
                        
                        # 式(30a) 和 (30b): 基于上一轮 F_best 计算的 gamma, phi, zeta
                        gamma_best = (abs(h_matrix[s, k, k])**2 * P_fixed[l, s, k]) / (noise + I_skl + 1e-12)
                        phi_best = gamma_best / (1 + gamma_best)
                        zeta_best = np.log2(1 + gamma_best) - phi_best * np.log2(gamma_best + 1e-12)
                        
                        # 构造需要带入 CVXPY 的替代函数项 (主要针对F_var相关的展开)
                        # 由于 CVXPY 重载了运算符，可以直接传入 F_var 获取解析表达式
                        I_skl_var = self._calculate_interference(P_fixed, F_var, h_matrix, s, k, l)
                        
                        log2_e = np.log2(np.e)
                        # 替代函数第一项（包含上一轮先验信息的常数乘子，仅保留对 F_var 呈线性的部分）
                        term1 = phi_best * np.log2(np.e * gamma_best + 1e-12) + zeta_best
                        # 替代函数第二项（干扰项的线性化惩罚）
                        term2_coeff = log2_e * phi_best * gamma_best / (abs(h_matrix[s, k, k])**2 * P_fixed[l, s, k] + 1e-12)
                        term2 = term2_coeff * (noise + I_skl_var)
                        
                        surrogate_R_sk += W * (term1 * F_var[s, k] - term2)
                    
                    # 式(32a)：每个 (s, k) 对应的传输数据包数量不能超过其信道容量（考虑参数 T0 / M0）
                    rate_sk_bound = surrogate_R_sk * (self.config.TIME_SLOT_DURATION / self.config.PACKET_SIZE)
                    # 约束32a
                    constraints += [X_var[s, k] <= rate_sk_bound]
            
            # utility_surrogate 表示论文最小化目标式19中q_sk*x_sk这一项，仅表示这一项是因为固定P、B时其他项为定值，最小化目标中仅剩余此项需要优化
            utility_surrogate = cp.sum(cp.multiply(Q_lengths, X_var))  
            
            # J_mp 即对应式(29)完整展开项，内部已包含 alpha 和 beta 相关惩罚
            objective = cp.Minimize(-utility_surrogate / self.config.L_0 + J_mp)
            
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.SCS)
                if F_var.value is not None:
                    F_best = F_var.value
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
                if F_best[s, k_idx] > 0.05:
                    F_quantized[s, k_idx] = 1.0
                    
        return F_quantized

    def solve_P_SCA(self, F_fixed, P_prev, B_fixed, h_matrix, g_matrix, Q_lengths):
        ''' Algorithm 3: SCA for Power '''
        P_var = cp.Variable((self.L, self.S, self.K), nonneg=True)
        constraints = []
        for s in range(self.S):
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
