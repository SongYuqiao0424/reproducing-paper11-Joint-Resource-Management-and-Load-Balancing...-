F_opt = algo.solvers.solve_F_MPMM(algo.F_prev, algo.P_prev, algo.B_prev, h_matrix, g_matrix, env.queue_lengths)
        _, P_opt, B_opt = bcd_optimization_placeholder(env, config, h_matrix, g_matrix, F_in=F_opt)
        algo.F_prev, algo.P_prev, algo.B_prev = F_opt, P_opt, B_opt