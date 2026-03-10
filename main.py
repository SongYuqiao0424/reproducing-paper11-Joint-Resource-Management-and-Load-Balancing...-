import os
import numpy as np
import time

from config import Config
from env.satellite_network import SatelliteNetworkEnv
from utils.plotter import plot_simulation_results, plot_beam_power_heatmap

def bcd_optimization_placeholder(env, config, h_matrix, g_matrix):
    '''
    这是一个用于替代完整 BCD(MPMM + SCA + 二次规划) 求解的启发式/随机占位函数。
    在 algorithms/solvers.py 尚未实现和引入之前，用来验证系统闭环运转。
    它将返回形状符合预期的 F, P, B 张量。
    '''
    S = config.NUM_SATELLITES
    K = config.NUM_CELLS
    L = config.NUM_FREQUENCY_SEGMENTS
    
    # 随机生成一个合理的跳波束策略图 F (S x K) - 每颗卫星随机在其覆盖小区(OMEGA_S)内选 Nb 个波束服务
    F_pattern = np.zeros((S, K))
    for s in range(S):
        available_cells = config.OMEGA_S[s]
        num_to_choose = min(config.NUM_BEAMS_PER_SAT, len(available_cells))
        if num_to_choose > 0:
            chosen_cells = np.random.choice(available_cells, num_to_choose, replace=False)
            F_pattern[s, chosen_cells] = 1.0
        
    # 随机生成符合功率上限的功率分配矩阵 P (L x S x K)
    P_matrix = np.random.uniform(6.0, 10.0, (L, S, K))
    # 仅向 F 限定的波束注入功率
    for s in range(S):
        for k in range(K):
            if F_pattern[s, k] == 0.0:
                P_matrix[:, s, k] = 0.0
            else:
                # 简单归一化，保证满足 MAX_POWER_PER_SAT 约束
                p_sum = np.sum(P_matrix[:, s, k])
                if p_sum > config.MAX_POWER_PER_SAT / config.NUM_BEAMS_PER_SAT:
                    scaling = (config.MAX_POWER_PER_SAT / config.NUM_BEAMS_PER_SAT) / p_sum
                    P_matrix[:, s, k] *= scaling
                    
    # 随机生成少量的负载转移张量 B (S x S x K)
    B_tensor = np.zeros((S, S, K))
    for s in range(S):
        for k in range(K):
            if F_pattern[s, k] == 1.0:  # 只有当 s 分配到了服务 k 的任务，才有可能需要把多余负载转给别人
                # 选择同属可服务第k个小区的 LEO 卫星集合 PHI_K 的其他卫星作为接收方
                phi_k = config.PHI_K[k]
                available_receivers = [x for x in phi_k if x != s]
                
                if available_receivers:  # 如果存在可用的接收卫星
                    r = np.random.choice(available_receivers)
                    val = np.random.uniform(0.0, 5.0)
                    B_tensor[s, r, k] = -val  # s给r发，s流出为负
                    B_tensor[r, s, k] = val   # r接收，r流入为正
    
    return F_pattern, P_matrix, B_tensor

def main():
    np.set_printoptions(precision=1, edgeitems=5, linewidth=120, suppress=True)
    print('===========================================================')
    print('  Starting Multi-Satellite Beam Hopping Simulation (Energy Min) ')
    print('===========================================================')
    
    # 1. 载入配置与环境
    config = Config()
    env = SatelliteNetworkEnv(config)
    
    print(f'[INFO] Number of Satellites: {config.NUM_SATELLITES}')
    print(f'[INFO] Number of Beams per Sat: {config.NUM_BEAMS_PER_SAT}')
    print(f'[INFO] Number of Ground Cells: {config.NUM_CELLS}')
    print(f'[INFO] Max Power limitation: {config.MAX_POWER_PER_SAT} W')
    print(f'[INFO] Simulation Slots: {config.MAX_TIME_SLOTS}')
    print('-----------------------------------------------------------')

    start_time = time.time()
    
    # 用于收集卫星0的历史功率矩阵以供热力图绘制
    p_history_sat0 = []
    
    # 2. 模拟时隙循环开始 (指标由 env.history_metrics 统一管理)
    for n in range(config.MAX_TIME_SLOTS):
        # (a) 更新与获取信道状态 (当前时隙的 H 与 G)
        h_matrix, g_matrix = env.channel_model.generate_random_channel_matrices()
        
        # (b) 进行联合求解策略计算：BCD 优化获取 F[n], P[n], B[n] 
        # TODO: 替换为 algorithms/solvers.py 中真实的 BCD 函数
        F_opt, P_opt, B_opt = bcd_optimization_placeholder(env, config, h_matrix, g_matrix)
        
        # (c) 执行动作并在环境中步进，产生延时与能耗表现
        step_metrics = env.step(F_opt, P_opt, B_opt)
        if n < 50:
            p_history_sat0.append(P_opt.copy())
        
        # 屏幕显示进度
        if (n + 1) % 100 == 0:
            avg_q = step_metrics['avg_queue']
            pwr = step_metrics['avg_power']
            tpt = step_metrics['throughput']
            drp = step_metrics['drop_rate']
            print(f'[Slot {n+1:4d} / {config.MAX_TIME_SLOTS}] Queue: {avg_q:.2f} pkts | Power: {pwr:.4f} W | Tput: {tpt:.2f} | Drop Rate: {drp*100:.2f}%')
            #print(f'Queue Lengths Matrix Sample:\n{env.queue_lengths}')

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
    print('[INFO] Done.')
    
if __name__ == '__main__':
    main()

