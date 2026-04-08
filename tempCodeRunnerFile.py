F_opt, _, _ = bcd_optimization_placeholder(env, config, h_matrix, g_matrix)
        P_opt = algo.solvers.solve_P_SCA(F_opt, algo.P_prev, algo.B_prev, h_matrix, g_matrix, env.queue_lengths)
        _, _, B_opt = bcd_optimization_placeholder(env, config, h_matrix, g_matrix, F_in=F_opt, P_in=P_opt)
        algo.F_prev, algo.P_prev, algo.B_prev = F_opt, P_opt, B_opt