import os
import numpy as np
from .solvers import OptimizationSolvers

class ProposedAlgorithm:
    '''
    打包论文核心的 Algorithm 1 (BCD Algorithm)
    '''
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.solvers = OptimizationSolvers(config, env)
        
        # 缓存上一时隙的结果，以便热启动加速收敛
        S = self.config.NUM_SATELLITES
        K = self.config.NUM_CELLS
        L = self.config.NUM_FREQUENCY_SEGMENTS
        
        self.F_prev = np.zeros((S, K))
        self.P_prev = np.full((L, S, K), self.config.MAX_POWER_PER_SAT / (L * self.config.NUM_BEAMS_PER_SAT))
        self.B_prev = np.zeros((S, S, K))
        
    def step(self, h_matrix, g_matrix, Q_lengths):
        '''
        计算单个时隙内的联合资源优化方案
        返回: (F_opt, P_opt, B_opt) 对应当前时隙的最优策略
        '''
        # 使用基于论文的 BCD 交替下降循环
        max_loops = getattr(self.config, 'MAX_BCD_LOOPS', 5)
        
        F_current = self.F_prev.copy()
        P_current = self.P_prev.copy()
        B_current = self.B_prev.copy()
        
        for _ in range(max_loops):
            # 1. 优化 波束跳频模式 F (使用 MPMM 算法)
            F_next = self.solvers.solve_F_MPMM(F_current, P_current, B_current, h_matrix, g_matrix, Q_lengths)
            
            # 2. 优化 功率分配与频段资源 P (使用 SCA 算法)
            P_next = self.solvers.solve_P_SCA(F_next, P_current, B_current, h_matrix, g_matrix, Q_lengths)
            
            # 3. 优化 负载均衡矩阵 B (使用 QP 二次规划算法)
            B_next = self.solvers.solve_B_QP(F_next, P_next, B_current, Q_lengths)
            
            F_current = F_next
            P_current = P_next
            B_current = B_next
            
        self.F_prev = F_current
        self.P_prev = P_current
        self.B_prev = B_current
        
        return F_current, P_current, B_current
