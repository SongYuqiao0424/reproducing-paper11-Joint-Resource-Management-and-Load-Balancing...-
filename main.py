import os
import numpy as np
import time
import shutil

from config import Config
from env.satellite_network import SatelliteNetworkEnv
from utils.plotter import (
    plot_simulation_results,
    plot_beam_power_heatmap,
    plot_beam_selection_heatmap,
    plot_sat0_exchange_matrix,
    plot_cell69_queue_and_transfer,
)
from algorithms.proposed_algo import ProposedAlgorithm

def bcd_optimization_placeholder(env, config, h_matrix, g_matrix, F_in=None, P_in=None, B_in=None):
    '''
    这是一个用于替代完整 BCD(MPMM + SCA + 二次规划) 求解的启发式/随机占位函数。
    在 algorithms/solvers.py 尚未实现和引入之前，用来验证系统闭环运转。
    当提供 F_in, P_in, B_in 时，将直接使用；否则基于逻辑进行生成（例如 P, B 根据生效的 F 限定生成）。
    '''
    S = config.NUM_SATELLITES
    K = config.NUM_CELLS
    L = config.NUM_FREQUENCY_SEGMENTS
    
    # 1. 确定 F_pattern: 若存在输入则直接复制，否则由于受限于区域覆盖，随机生成
    if F_in is not None:
        F_pattern = F_in.copy()
    else:
        F_pattern = np.zeros((S, K))
        for s in range(S):
            available_cells = config.OMEGA_S[s]
            num_to_choose = min(config.NUM_BEAMS_PER_SAT, len(available_cells))
            if num_to_choose > 0:
                chosen_cells = np.random.choice(available_cells, num_to_choose, replace=False)
                F_pattern[s, chosen_cells] = 1.0
        
    # 2. 确定 P_matrix: 若存在输入则直接复制，否则给生效的 F 随机分配并归一化
    if P_in is not None:
        P_matrix = P_in.copy()
    else:
        P_matrix = np.random.uniform(6.0, 10.0, (L, S, K))
        # 仅向当前 F_pattern 限定的波束注入功率
        for s in range(S):
            for k in range(K):
                if F_pattern[s, k] == 0.0:
                    P_matrix[:, s, k] = 0.0
                else:
                    p_sum = np.sum(P_matrix[:, s, k])
                    if p_sum > config.MAX_POWER_PER_SAT / config.NUM_BEAMS_PER_SAT:
                        scaling = (config.MAX_POWER_PER_SAT / config.NUM_BEAMS_PER_SAT) / p_sum
                        P_matrix[:, s, k] *= scaling
                        
    # 3. 确定 B_tensor: 若存在输入则直接复制，否则在生效的 F 上构建负载均衡跳转
    if B_in is not None:
        B_tensor = B_in.copy()
    else:
        B_tensor = np.zeros((S, S, K))
        for s in range(S):
            for k in range(K):
                if F_pattern[s, k] == 1.0: 
                    phi_k = config.PHI_K[k]
                    available_receivers = [x for x in phi_k if x != s]
                    if available_receivers:
                        r = np.random.choice(available_receivers)
                        val = np.random.uniform(0.0, 5.0)
                        B_tensor[s, r, k] = -val
                        B_tensor[r, s, k] = val
    
    return F_pattern, P_matrix, B_tensor

def main():
    np.set_printoptions(precision=1, edgeitems=5, linewidth=120, suppress=True)
    print('===========================================================')
    print('  Starting Multi-Satellite Beam Hopping Simulation (Energy Min) ')
    print('===========================================================')
    
    # 1. 载入配置与环境
    config = Config()
    env = SatelliteNetworkEnv(config)
    
    # 实例化提议的BCD优化算法
    algo = ProposedAlgorithm(config, env)
    # 初始化 F_prev, P_prev, B_prev 以供算法使用
    h_matrix, g_matrix = env.channel_model.generate_random_channel_matrices()
    algo.F_prev, algo.P_prev, algo.B_prev = bcd_optimization_placeholder(env, config, h_matrix, g_matrix)
    
    print(f'[INFO] Number of Satellites: {config.NUM_SATELLITES}')
    print(f'[INFO] Number of Beams per Sat: {config.NUM_BEAMS_PER_SAT}')
    print(f'[INFO] Number of Ground Cells: {config.NUM_CELLS}')
    print(f'[INFO] Max Power limitation: {config.MAX_POWER_PER_SAT} W')
    print(f'[INFO] Simulation Slots: {config.MAX_TIME_SLOTS}')
    print('-----------------------------------------------------------')

    # 先展示卫星0在其覆盖小区上与其他卫星的可交换关系图
    # plot_sat0_exchange_matrix(config)

    start_time = time.time()
    
    # 用于收集卫星0的历史功率矩阵以供热力图绘制
    p_history_sat0 = []
    f_history_sat0 = []
    q0_cell69_hist = []
    q1_cell69_hist = []
    b01_cell69_hist = []
    sat0_cells = sorted(config.OMEGA_S.get(0, []))
    
    # 2. 模拟时隙循环开始 (指标由 env.history_metrics 统一管理)
    for n in range(config.MAX_TIME_SLOTS):
        # (a) 更新与获取信道状态 (当前时隙的 H 与 G)
        h_matrix, g_matrix = env.channel_model.generate_random_channel_matrices()
        
        # (b) 进行联合求解策略计算：BCD 优化获取 F[n], P[n], B[n] 
        # 使用随机占位函数替代完整 BCD 求解
        F_opt, P_opt, B_opt = bcd_optimization_placeholder(env, config, h_matrix, g_matrix)

        # 使用真实的 BCD 交替优化算法求解
        # F_opt, P_opt, B_opt = algo.step(h_matrix, g_matrix, env.queue_lengths)

        # 单独测试 MPMM 算法对 F 的优化效果
        # F_opt = algo.solvers.solve_F_MPMM(algo.F_prev, algo.P_prev, algo.B_prev, h_matrix, g_matrix, env.queue_lengths)
        # _, P_opt, B_opt = bcd_optimization_placeholder(env, config, h_matrix, g_matrix, F_in=F_opt)
        # algo.F_prev, algo.P_prev, algo.B_prev = F_opt, P_opt, B_opt
        
        # 单独测试 MPMM 算法对 P 的优化效果
        # F_opt, _, _ = bcd_optimization_placeholder(env, config, h_matrix, g_matrix)
        # P_opt = algo.solvers.solve_P_SCA(F_opt, algo.P_prev, algo.B_prev, h_matrix, g_matrix, env.queue_lengths)
        # _, _, B_opt = bcd_optimization_placeholder(env, config, h_matrix, g_matrix, F_in=F_opt, P_in=P_opt)
        # algo.F_prev, algo.P_prev, algo.B_prev = F_opt, P_opt, B_opt

        # 单独测试 MPMM 算法对 B 的优化效果
        # F_opt, P_opt, _ = bcd_optimization_placeholder(env, config, h_matrix, g_matrix)
        # B_opt = algo.solvers.solve_B_QP(F_opt, P_opt, algo.B_prev, h_matrix, env.queue_lengths)
        # algo.F_prev, algo.P_prev, algo.B_prev = F_opt, P_opt, B_opt


        # (c) 执行动作并在环境中步进，产生延时与能耗表现
        step_metrics = env.step(F_opt, P_opt, B_opt)

        # 记录小区69上 Sat0/Sat1 队列长度与 Sat0->Sat1 负载均衡传输量，用于绘制负载均衡校验图
        target_cell = 69
        if config.NUM_CELLS > target_cell:
            q0_cell69_hist.append(env.queue_lengths[0, target_cell] if config.NUM_SATELLITES > 0 else 0.0)
            q1_cell69_hist.append(env.queue_lengths[1, target_cell] if config.NUM_SATELLITES > 1 else 0.0)
            b01_cell69_hist.append(B_opt[0, 1, target_cell] if config.NUM_SATELLITES > 1 else 0.0)

        # 记录卫星0的功率分配和波束选择，用于绘制热力图
        if n < 50:
            p_history_sat0.append(P_opt.copy())
            f_history_sat0.append(F_opt.copy())
        
        # 屏幕显示进度
        if (n + 1) % 1 == 0:
            avg_q = step_metrics['avg_queue']
            pwr = step_metrics['avg_power']
            tpt = step_metrics['throughput']
            drp = step_metrics['drop_rate']
            print(f'[Slot {n+1:4d} / {config.MAX_TIME_SLOTS}] Queue: {avg_q:.2f} pkts | Power: {pwr:.4f} W | Tput: {tpt:.2f} | Drop Rate: {drp*100:.2f}%')
            
            # --- 新增：仅打印本时隙卫星0选择了哪些小区及其对应的 F_opt 分值 ---
            if sat0_cells:
                sat0_scores = F_opt[0, sat0_cells]
                top_n = min(4, len(sat0_cells))
                selected_local = np.argsort(sat0_scores)[-top_n:][::-1]
                selected_indices = sorted([sat0_cells[i] for i in selected_local])
            else:
                selected_indices = []
            details = [f"C{idx}({F_opt[0, idx]:.2f})" for idx in selected_indices]
            print(f"    Sat0 Beam Allocation -> [{', '.join(details)}]")
            
            # 打印卫星0对每个小区分别的功率 (仅对频段L求和)
            sat0_power = np.sum(P_opt[:, 0, :], axis=0)
            # 加上编号方便对比，这里仅展示分配功率大于0的小区以防太长
            active_power_strs = [f"C{k}:{sat0_power[k]:.2f}W" for k in sat0_cells if sat0_power[k] > 0.001]
            if not active_power_strs:
                active_power_strs = ["All 0W"]
            print(f"    Sat0 Power Alloc -> [{', '.join(active_power_strs)}]")

             # 打印卫星0覆盖小区上的队列长度，带上小区号
            q_strs = [f"C{k}:{env.queue_lengths[0, k]:.1f}" for k in sat0_cells]
            print(f"    Sat0 Queue Lengths -> [{', '.join(q_strs)}]")

            # 仅打印卫星0与其他卫星共同覆盖的小区：
            # Cnum:S0_queuelength_on/off;Snum1_queuelength_on/off;Snum2_...;load_balance
            shared_lb_items = []
            for k in sat0_cells:
                shared_sats = sorted([s for s in config.PHI_K.get(k, []) if s != 0])
                if not shared_sats:
                    continue

                s0_q = env.queue_lengths[0, k]
                s0_state = 'on' if F_opt[0, k] > 0.5 else 'off'
                sat_parts = [f"S0_{s0_q:.1f}_{s0_state}"]
                for s in shared_sats:
                    s_q = env.queue_lengths[s, k]
                    s_state = 'on' if F_opt[s, k] > 0.5 else 'off'
                    sat_parts.append(f"S{s}_{s_q:.1f}_{s_state}")

                d_0k = sum(B_opt[r, 0, k] for r in config.PHI_K.get(k, []))
                shared_lb_items.append(f"C{k}:{';'.join(sat_parts)};{d_0k:+.2f}")

            print("    Sat0 Shared Load Balance ->")
            if shared_lb_items:
                term_width = shutil.get_terminal_size(fallback=(160, 24)).columns
                half_width = max(40, term_width // 2)
                i = 0
                while i < len(shared_lb_items):
                    left_item = shared_lb_items[i]

                    # 若左侧内容超过半宽，则本行只打印一条
                    if len(left_item) > half_width:
                        print(left_item)
                        i += 1
                        continue
                    # 尝试放置右侧内容；若右侧超过半宽则本行仅打印左侧
                    if i + 1 < len(shared_lb_items):
                        right_item = shared_lb_items[i + 1]
                        if len(right_item) <= half_width:
                            print(f"{left_item.ljust(half_width)}{right_item}")
                            i += 2
                            continue
                    print(left_item)
                    i += 1
            else:
                print("        No shared cells with other satellites")

            print("")

    # 3. 运行结束，输出总体仿真指标
    elapsed_time = time.time() - start_time
    print('-----------------------------------------------------------')
    print(f'Simulation Finished in {elapsed_time:.2f} seconds.')
    print(f'[Result] Overall Avg Queue Length: {np.mean(env.history_metrics["avg_queue"]):.2f} pkts')
    print(f'[Result] Overall Avg Power: {np.mean(env.history_metrics["avg_power"]):.2f} W')
    print(f'[Result] Overall Sum Throughput: {np.sum(env.history_metrics["total_throughput"]):.2f} pkts')
    print(f'[Result] Overall Drop Rate (Mean): {np.mean(env.history_metrics["drop_rate"])*100:.2f} %')
    print('===========================================================')
    
    print('[INFO] Generating performance plots in 仿真结果/ ...')
    plot_simulation_results(env.history_metrics, config)
    if len(p_history_sat0) > 0:
        plot_beam_power_heatmap(p_history_sat0, config)
        plot_beam_selection_heatmap(f_history_sat0, config)
    if len(q0_cell69_hist) > 0:
        plot_cell69_queue_and_transfer(q0_cell69_hist, q1_cell69_hist, b01_cell69_hist)
    print('[INFO] Done.')
    
if __name__ == '__main__':
    main()


