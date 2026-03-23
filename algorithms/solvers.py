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
        # 提高了惩罚项J的系数α、β、ρ的值,避免陷入0.5陷阱,确保F值接近0/1；
        base_penalty = 1.0  # 正常比例尺
        alpha = np.ones((self.S, self.K)) * (base_penalty * 0.1)     # 拉格朗日乘子系数
        beta = base_penalty * 0.5                                    # 惩罚函数系数
        rho = 1.5                                                    # 惩罚函数更新速率加快
        Theta_rounds = getattr(self.config, 'MPMM_THETA_ROUNDS', 5)
        Psi_rounds = getattr(self.config, 'MPMM_PSI_ROUNDS', 5)
        
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
        for theta in range(Theta_rounds):
            F_ref = F_best.copy()
            for psi in range(Psi_rounds):
                # 约束13c
                constraints = [F_var <= valid_mask]
                for s in range(self.S):
                    # 约束13d
                    constraints += [cp.sum(F_var[s, :]) <= self.config.NUM_BEAMS_PER_SAT]
                for coeff in GSO_coeffs:
                    # 约束13g
                    constraints += [cp.sum(cp.multiply(coeff, F_var)) <= Z_w_limit]

                f_surrogate = cp.multiply(1 - 2 * F_ref, F_var) + F_ref**2
                J_mp = cp.sum(cp.multiply(alpha, f_surrogate)) + beta * cp.sum(cp.square(f_surrogate))

                # 引入真实传输变量 X_var (体现原论文式31中的辅助变量转换，以及式32a中对每个链路的约束)
                X_var = cp.Variable((self.S, self.K))
                # 约束 13l
                constraints += [X_var <= Q_temp]

                # 式1的I_skl计算
                I_fixed = np.zeros((self.S, self.K, self.L))
                for k in range(self.K):
                    # 直接计算小区k受到PHI_K内所有卫星的信号总和，包括了干扰和有用信号，再减去有用信号得到总干扰
                    total_I_k = np.zeros(self.L)
                    for s_idx in self.config.PHI_K[k]:
                        for k_idx in range(self.K):
                            total_I_k += HP[k, s_idx, k_idx, :] * F_ref[s_idx, k_idx]
                    for s in self.config.PHI_K[k]:
                        I_fixed[s, k, :] = np.maximum(0, total_I_k - HP[k, s, k, :] * F_ref[s, k])

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
                if theta == 0 and psi == 0:
                    print(f"        [Debug] Term1_sum max: {np.max(Term1_sum):.2e}, Noise_sum max: {np.max(Noise_sum):.2e}")

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

                # utility_surrogate 表示论文最小化目标式19中q_sk*x_sk这一项，仅表示这一项是因为固定P、B时其他项为定值，最小化目标中仅剩余此项需要优化
                utility_surrogate = cp.sum(cp.multiply(Q_lengths, X_var))

                # J_mp 即对应式(29)完整展开项，内部已包含 alpha 和 beta 相关惩罚
                # 修改主优化目标函数utility_surrogate的均衡系数，让目标函数处于合理的数值范围，这是导致 solver 不精确的根本原因
                # objective = cp.Minimize(-utility_surrogate / np.max([1.0, np.max(Q_lengths)]) + J_mp)
                objective = cp.Minimize(-utility_surrogate / self.config.L_0 + J_mp)

                prob = cp.Problem(objective, constraints)
                try:
                    # 迭代次数影响求解的精度，迭代次数较多时，精度较高，但仿真时间会过长;
                    # 迭代次数较少时,可能未到达最优解,比如此时每个卫星的波束激活量sum(F_best[s,:])不小于等于NUM_BEAMS_PER_SAT(不满足约束13d)
                    # 也有可能单个波束选择值F_best[s,k]未迭代至0/1;但是只要求解到可行解附近时即可判断出应该激活的卫星波束.
                    prob.solve(solver=cp.SCS, warm_start=True, max_iters=10000, eps=1e-4)
                    if F_var.value is not None:
                        F_ref = F_var.value
                        x_val = X_var.value
                        x_sum = np.sum(x_val) if x_val is not None else -10
                        x_max = np.max(x_val) if x_val is not None else -10
                        # print(f"        [Theta {theta} Psi {psi}] Current F_ref sum: {np.sum(F_ref):.2f}, Max Val: {np.max(F_ref):.2f} | X_var sum: {x_sum:.4f}, Max: {x_max:.4f}")
                        # print(f"        [Theta {theta} Psi {psi}] Sat0 F_ref: {np.array2string(np.round(F_ref[0, :19],2), separator=',', max_line_width=200)}")
                        # print(f"        [Theta {theta} Psi {psi}] Sat0 X_var: {np.array2string(np.round(x_val[0, :19],2), separator=',', max_line_width=200) if x_val is not None else 'None'}")
                    else:
                        break
                except Exception:
                    break

            F_best = F_ref
            alpha += 2 * beta * F_best * (1 - F_best)
            beta *= rho

        F_quantized = np.zeros((self.S, self.K))
        for s in range(self.S):
            top_k_indices = np.argsort(F_best[s, :])[-self.config.NUM_BEAMS_PER_SAT:]
            for k_idx in top_k_indices:
                F_quantized[s, k_idx] = 1.0
                    
        return F_quantized

    def solve_P_SCA(self, F_fixed, P_prev, B_fixed, h_matrix, g_matrix, Q_lengths):
        ''' Algorithm 3: SCA for Power '''
        Xi_rounds = getattr(self.config, 'SCA_XI_ROUNDS', 5)
        P_best = P_prev.copy()
        sk_size = self.S * self.K
        P_best_flat = P_best.reshape(self.L, sk_size)

        # 约束13l：考虑固定B后的临时队列上限 Q_temp = max(0, Q + d)
        d_sk = np.zeros((self.S, self.K))
        for s_idx in range(self.S):
            for k_idx in range(self.K):
                d_sk[s_idx, k_idx] = sum([B_fixed[r, s_idx, k_idx] for r in self.config.PHI_K[k_idx]])
        Q_temp = np.maximum(0.0, Q_lengths + d_sk)

        noise = self.env.channel_model.noise_power * self.config.BANDWIDTH_PER_SEGMENT
        log2_e = np.log2(np.e)
        time_scale = self.config.TIME_SLOT_DURATION / self.config.PACKET_SIZE
        W_band = self.config.BANDWIDTH_PER_SEGMENT

        # 计算有用信号的系数矩阵
        h_self_sq = np.abs(h_matrix[:, np.arange(self.K), np.arange(self.K)])**2
        active_link_mask = (F_fixed > 0) & (h_self_sq > 1e-18)

        # 第一轮Xi采用虚拟参考功率，避免新激活链路在P_prev=0时线性化退化
        P_virtual_ref = np.zeros_like(P_best)
        for s in range(self.S):
            # 计算当前卫星s的激活波束数量
            active_k_idx = np.where(active_link_mask[s, :])[0]
            active_count = len(active_k_idx)
            if active_count == 0:
                continue
            # 均分初始参考功率
            p_seed = self.config.MAX_POWER_PER_SAT / (active_count * self.L)
            P_virtual_ref[:, s, active_k_idx] = p_seed
        P_virtual_ref_flat = P_virtual_ref.reshape(self.L, sk_size)

        # 预计算接收信号的系数矩阵 base_coeff_by_k[k, s*K+k_idx],包括有用信号和干扰信号的系数
        base_coeff_by_k = np.zeros((self.K, sk_size))
        for k in range(self.K):
            # 每一行表示小区k收到的信号系数,顺序为(s_idx * K + k_idx)
            row = base_coeff_by_k[k]
            for s_idx in self.config.PHI_K[k]:
                # 计算小区k收到s_idx卫星到各小区波束的信号系数,长度为K
                coeff_slice = np.abs(h_matrix[s_idx, k, :])**2 * F_fixed[s_idx, :]
                # 将K个信号系数写入对应位置上
                row[s_idx * self.K:(s_idx + 1) * self.K] = coeff_slice

        # 预计算每个(s,k)的有用信号系数矩阵，用于后续从总接收信号矩阵中剔除
        self_coeff = np.zeros((self.S, self.K))
        for s in range(self.S):
            for k in range(self.K):
                self_coeff[s, k] = base_coeff_by_k[k, s * self.K + k]

        # 预计算GSO干扰系数向量
        Z_w_limit = 10 ** (self.config.Z_MAX_DBW / 10.0)
        K_G = getattr(self.config, 'K_G', [])
        L_K = getattr(self.config, 'L_K', {})
        gso_constraints_data = []
        for k_g in K_G:
            # 分别存储每个GSO小区k_g的重叠频段列表和对应的干扰系数向量
            overlap_bands = list(L_K.get(k_g, range(self.L)))
            coeff_vec = np.zeros(sk_size)
            for s in range(self.S):
                # 对于小区k_g，依次计算卫星s到各小区的波束的干扰系数
                coeff_vec[s * self.K:(s + 1) * self.K] = np.abs(g_matrix[s, k_g, :])**2 * F_fixed[s, :]
            gso_constraints_data.append((overlap_bands, coeff_vec))

        for xi in range(Xi_rounds):
            # 约束13e: 使用二维变量减少CVXPY高维表达式构建开销
            P_var = cp.Variable((self.L, sk_size), nonneg=True)
            X_var = cp.Variable((self.S, self.K), nonneg=True)
            constraints = []

            # 仅第一轮使用虚拟参考点，后续轮次使用上一轮解
            P_ref_flat = P_virtual_ref_flat if xi == 0 else P_best_flat

            for s in range(self.S):
                # 约束13f
                constraints += [cp.sum(P_var[:, s * self.K:(s + 1) * self.K]) <= self.config.MAX_POWER_PER_SAT]

            for overlap_bands, coeff_vec in gso_constraints_data:
                # 约束13g：依次取重叠频段，计算所有s_k波束的功率与干扰系数乘积，并累加求和
                constraints += [cp.sum(cp.multiply(P_var[overlap_bands, :], coeff_vec[None, :])) <= Z_w_limit]

            # 约束13l
            constraints += [X_var <= Q_temp]

            # 预构建 I_base(k,l) = sum(base_coeff_by_k[k] * P_var[l,:])，I_base(k,l) 表示频段l上，小区k受到的总信号功率，包括有用信号和干扰信号，后续在计算I_var时再剔除有用信号部分
            I_var_base = []
            for k in range(self.K):
                # 记录小区k在每个频段l上的总接收功率
                i_var_row = []
                for l in range(self.L):
                    i_var_row.append(cp.sum(cp.multiply(base_coeff_by_k[k], P_var[l, :])))
                I_var_base.append(i_var_row)

            # 基于式(35)(36)构造 R^P_{s,k}(P; P_hat), 其中 P_hat = 当前迭代点 P_best
            for s in range(self.S):
                for k in range(self.K):
                    idx_sk = s * self.K + k
                    if not active_link_mask[s, k]:
                        # 约束13k的一部分：对于非激活链路，直接约束X_var为0
                        constraints += [X_var[s, k] == 0]
                        constraints += [P_var[:, idx_sk] == 0]
                        continue
                    
                    coeff_self = self_coeff[s, k]
                    h_sq = h_self_sq[s, k]
                    surrogate_R_sk = 0

                    for l in range(self.L):
                        # I_{s,k,l}(P_hat)，先计算小区k在频段l上的总接收功率，再减去来自卫星s的有用信号
                        I_hat_base = np.dot(base_coeff_by_k[k], P_ref_flat[l, :])
                        I_hat = max(0.0, I_hat_base - coeff_self * P_ref_flat[l, idx_sk])

                        # 式(35): eta 在 P_hat 处的取值；phi、zeta 由 gamma_hat 计算
                        eta_hat = (h_sq * F_fixed[s, k]) / (noise + I_hat + 1e-15)
                        gamma_hat = eta_hat * P_ref_flat[l, idx_sk]
                        phi_hat = gamma_hat / (1.0 + gamma_hat)
                        zeta_hat = np.log2(1.0 + gamma_hat) - phi_hat * np.log2(gamma_hat + 1e-15)

                        # I_{s,k,l}(P): 关于 P_var 的仿射表达
                        I_var = I_var_base[k][l] - coeff_self * P_var[l, idx_sk]

                        # 式(36): 单频段近似速率项
                        term_log = phi_hat * cp.log(np.e * eta_hat * P_var[l, idx_sk] + 1e-15) / np.log(2)
                        term_affine = log2_e * (phi_hat * eta_hat) * ((noise + I_var) / (h_sq + 1e-15))
                        surrogate_R_l = W_band * (F_fixed[s, k] * (term_log + zeta_hat) - term_affine)
                        surrogate_R_sk += surrogate_R_l

                    # 约束37a: x_{s,k} * M0 <= R^P_{s,k}(P; P_hat) * T0
                    constraints += [X_var[s, k] <= surrogate_R_sk * time_scale]

            # 漂移项使用辅助变量 x_{s,k}
            rate_expr = cp.sum(cp.multiply(Q_lengths, X_var))
            energy_expr = cp.sum(P_var) * self.config.TIME_SLOT_DURATION

            # rate_expr 表示论文最小化目标式19第一项中q_sk*x_sk这一项，仅表示这一项是因为固定F、B时其他项为定值，最小化目标中仅剩余此项需要优化
            # energy_expr 表示论文最小化目标式19第二项中卫星与地面波束能量消耗项，仅表示这一项是因为固定F、B时星间链路耗能为定值
            #objective = cp.Minimize(-rate_expr / self.config.L_0 + (self.config.V / self.config.E_0) * energy_expr)
            objective = cp.Minimize(-rate_expr / self.config.L_0)
            prob = cp.Problem(objective, constraints)
            try:
                cvx_verbose = getattr(self.config, 'CVX_VERBOSE', True)
                scs_max_iters = getattr(self.config, 'SCS_MAX_ITERS', 10000)
                scs_eps = getattr(self.config, 'SCS_EPS', 1e-4)

                prob.solve(
                    solver=cp.SCS,
                    warm_start=True,
                    max_iters=scs_max_iters,
                    eps=scs_eps,
                    # verbose=cvx_verbose
                )
                status = prob.status
                print(f"        [P-SCA][SCS-1] prob.status = {status}")

                # # 若仍是不精确解，先用更强SCS参数重试一次
                # if status == cp.OPTIMAL_INACCURATE:
                #     print("        [P-SCA][Warn] SCS首次求解为optimal_inaccurate，尝试增强参数重试。")
                #     prob.solve(
                #         solver=cp.SCS,
                #         warm_start=True,
                #         max_iters=max(80000, int(scs_max_iters * 1.5)),
                #         eps=min(scs_eps, 5e-5),
                #         verbose=cvx_verbose
                #     )
                #     status = prob.status
                #     print(f"        [P-SCA][SCS-2] prob.status = {status}")

                # # 若SCS两次后仍不精确，尝试CLARABEL（若已安装）
                # if status == cp.OPTIMAL_INACCURATE and "CLARABEL" in cp.installed_solvers():
                #     print("        [P-SCA][Warn] SCS仍不精确，尝试CLARABEL求解。")
                #     prob.solve(solver=cp.CLARABEL, verbose=cvx_verbose)
                #     status = prob.status
                #     print(f"        [P-SCA][CLARABEL] prob.status = {status}")

                if status == cp.OPTIMAL_INACCURATE:
                    print("        [P-SCA][Warn] final status=optimal_inaccurate：结果可用但需谨慎。")
            except Exception:
                break

            if P_var.value is not None:
                P_best_flat = np.clip(P_var.value, 0, None)
                P_best = P_best_flat.reshape(self.L, self.S, self.K)
            else:
                break
        
        # 将非激活链路的功率置0
        # P_best *= (F_fixed > 0)[None, :, :]
            
        return P_best

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

