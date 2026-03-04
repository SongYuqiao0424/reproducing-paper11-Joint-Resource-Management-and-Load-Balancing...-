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
        for s_idx in range(self.S):
            for k_idx in range(self.K):
                if not (s_idx == s and k_idx == k):
                    interference += abs(h_matrix[s_idx, k, k_idx])**2 * P_matrix[l, s_idx, k_idx] * F_pattern[s_idx, k_idx]
        return interference

    def solve_F_MPMM(self, F_prev, P_fixed, B_fixed, h_matrix, g_matrix, Q_lengths):
        ''' Algorithm 2: MPMM for F Matrix '''
        F_var = cp.Variable((self.S, self.K), nonneg=True)
        alpha = np.ones((self.S, self.K)) * 0.1
        beta = 0.5
        rho = 1.1
        Theta_rounds = getattr(self.config, 'MPMM_THETA_ROUNDS', 5)
        
        F_best = F_prev.copy()
        for _ in range(Theta_rounds):
            constraints = [F_var <= 1.0]
            for s in range(self.S):
                constraints += [cp.sum(F_var[s, :]) <= self.config.NUM_BEAMS_PER_SAT]
                
            J_mp = cp.sum(cp.multiply(alpha, (1 - 2*F_best)*F_var))
            
            expected_rates = np.zeros((self.S, self.K))
            for s in range(self.S):
                for k in range(self.K):
                    R_val = 0
                    for l in range(self.L):
                        I_skl = self._calculate_interference(P_fixed, F_best, h_matrix, s, k, l)
                        noise = self.env.channel_model.noise_power * self.config.BANDWIDTH_PER_SEGMENT
                        sinr = (abs(h_matrix[s, k, k])**2 * P_fixed[l, s, k]) / (noise + I_skl + 1e-12)
                        R_val += self.config.BANDWIDTH_PER_SEGMENT * np.log2(1 + sinr)
                    expected_rates[s, k] = R_val * (self.config.TIME_SLOT_DURATION / self.config.PACKET_SIZE)
                    
            utility = cp.sum(cp.multiply(Q_lengths, cp.multiply(expected_rates, F_var))) 
            objective = cp.Minimize(-utility / self.config.L_0 + beta * J_mp)
            
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
            for l in overlap_bands:
                interference_to_gso = 0
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
