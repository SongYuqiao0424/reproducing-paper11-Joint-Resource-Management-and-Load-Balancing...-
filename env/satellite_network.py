import numpy as np
from .channel_model import ChannelModel

class SatelliteNetworkEnv:
    def __init__(self, config):
        self.config = config
        self.channel_model = ChannelModel(config)
        
        # 队列状态矩阵 Q[n]: 大小为 |S| x |K|
        self.queue_lengths = np.zeros((self.config.NUM_SATELLITES, self.config.NUM_CELLS))
        
        # 当前用户数据包到达率 (每个时隙漂移)
        self.current_arrival_rates = np.random.uniform(
            self.config.ARRIVAL_RATE_MIN, 
            self.config.ARRIVAL_RATE_MAX, 
            self.config.NUM_CELLS
        )
        
        self.current_time_slot = 0
        
        # 存储历史指标 (能耗，队列长度，吞吐量等) 用于后续绘图
        self.history_metrics = {
            'avg_queue': [],
            'avg_power': [],
            'total_throughput': [],
            'drop_rate': []
        }

    def _drift_arrival_rates(self):
        '''每隔一段时间，小区的平均需求发生缓变飘移分布 (对应论文中的描述)'''
        drift = np.random.uniform(-10e6, 10e6, self.config.NUM_CELLS)
        self.current_arrival_rates += drift
        
        # 保证在 min 和 max 的边界之间
        self.current_arrival_rates = np.clip(
            self.current_arrival_rates, 
            self.config.ARRIVAL_RATE_MIN, 
            self.config.ARRIVAL_RATE_MAX
        )

    def generate_arrivals(self):
        '''
        根据当前泊松率生成每个小区在这 10ms (T0) 产生的新数据包总到达数 a_{s,k}[n]
        (在这里简单假设所有到达请求均匀分布给 |S| 个可服务它的卫星，或由某一个主卫星接受)
        '''
        S = self.config.NUM_SATELLITES
        K = self.config.NUM_CELLS
        
        # 将 Mbps 转化为这个时隙内的数据包个数
        # Mbps -> 10^6 bits / s. 
        # lambda_pkts = (Rate(bits/s) * T0) / M0
        lambda_list = (self.current_arrival_rates * self.config.TIME_SLOT_DURATION) / self.config.PACKET_SIZE
        
        # 对每个小区生成泊松分布的到达包数
        arrived_pkts_for_cells = np.random.poisson(lambda_list)
        
        # 构建并分配到所属服务卫星，分配策略：平均分配排队
        # a_sk_n表示arrived卫星s服务小区k的包数
        a_sk_n = np.zeros((S, K))
        for k in range(K):
            # 均匀分发请求至所有可覆盖该小区的卫星集合 Φ(k)
            phi_k = self.config.PHI_K[k]
            num_sats = len(phi_k)
            
            if num_sats > 0:
                pkts_per_sat = arrived_pkts_for_cells[k] // num_sats
                remainder = arrived_pkts_for_cells[k] % num_sats
    
                for idx, s in enumerate(phi_k):
                    a_sk_n[s, k] = pkts_per_sat
                    if idx < remainder: # 分配余数
                        a_sk_n[s, k] += 1
                    
        return a_sk_n

    def step(self, F_pattern, P_matrix, B_tensor, h_matrix=None):
        """
        按照正确的时隙序列完成调度更新：
        1. 记录此时隙的初始队列
        2. 负载均衡：根据B调整队列 (Q_temp = Q_init + D)
        3. 状态传输：根据F、P与Q_temp计算此时隙真实传输 x
        4. 时隙尾声：生成此时隙内发生的最新请求到达 A
        5. 计算下时隙初始队列：Q_next = Q_init + D - X + A
        """
        S = self.config.NUM_SATELLITES
        K = self.config.NUM_CELLS
        
        self.current_time_slot += 1
        
        # 0. 锁定此时隙操作的初始基准队列
        initial_q = self.queue_lengths.copy()
        
        # 1. 根据星间链路矩阵 B 计算此时隙的负载均衡变动 d_{s,k}
        d_sk_n = np.zeros((S, K))
        for s in range(S):
            for k in range(K):
                # B_tensor 已经是在优化中包含了方向符号的张量(b_{r,s} = -b_{s,r})，因此正负自带，无需二次相减
                phi_k = self.config.PHI_K[k]
                d_sk_n[s, k] = sum([B_tensor[r, s, k] for r in phi_k])
        
        # 计算负载均衡后的中间态队列 (作为本次发射的发包容量约束)
        temp_balanced_q = initial_q + d_sk_n
        temp_balanced_q = np.maximum(0.0, temp_balanced_q)
        
        # 2. 按SINR计算本时隙有效发包数目x_{s,k}速率上限R_{s,k}
        # 若外部未传入本时隙信道，则回退到当前几何信道生成
        # if h_matrix is None:
        #     h_matrix, _ = self.channel_model.generate_random_channel_matrices()

        noise = self.channel_model.noise_power * self.config.BANDWIDTH_PER_SEGMENT
        time_scale = self.config.TIME_SLOT_DURATION / self.config.PACKET_SIZE
        W_band = self.config.BANDWIDTH_PER_SEGMENT

        x_sk_n = np.zeros((S, K))

        # # P*100计算每个链路的速率上限
        # R_pkts_cap = np.zeros((S, K))
        # for s_idx in range(S):
        #     for k_idx in range(K):
        #         if F_pattern[s_idx, k_idx] > 0:
        #             total_power_sk = np.sum(P_matrix[:, s_idx, k_idx])
        #             R_pkts_cap[s_idx, k_idx] = total_power_sk * 100
        # R_pkts_cap = R_pkts_cap * F_pattern

        # 实际SINR计算每个链路的速率上限
        R_pkts_cap = np.zeros((S, K))
        for s in range(S):
            for k in range(K):
                if F_pattern[s, k] <= 0:
                    continue

                rate_sk_bps = 0.0
                for l in range(self.config.NUM_FREQUENCY_SEGMENTS):
                    signal = (abs(h_matrix[s, k, k]) ** 2) * P_matrix[l, s, k] * F_pattern[s, k]

                    interference = 0.0
                    for s_idx in self.config.PHI_K[k]:
                        for j in range(K):
                            if s_idx == s and j == k:
                                continue
                            interference += (abs(h_matrix[s_idx, k, j]) ** 2) * P_matrix[l, s_idx, j] * F_pattern[s_idx, j]

                    sinr = signal / (noise + interference + 1e-12)
                    rate_sk_bps += W_band * np.log2(1.0 + sinr)

                R_pkts_cap[s, k] = rate_sk_bps * time_scale
        
        for s in range(S):
            for k in range(K):
                # 真实传输量严格遭受"均衡后队列可用数据包量"的约束
                x_sk_n[s, k] = min(R_pkts_cap[s, k], temp_balanced_q[s, k])
                x_sk_n[s, k] = max(0.0, x_sk_n[s, k])
        
        # 3. 计算此时隙的新到达数据包量 a_{s,k}
        if self.current_time_slot % self.config.DEMAND_DRIFT_STEPS == 0:
            self._drift_arrival_rates()
        a_sk_n = self.generate_arrivals()
        
        # 4. 根据 时隙初始队列、负载均衡变更、传输量、新到到达量 更新出下一时隙队列
        total_dropped = 0.0
        for s in range(S):
            for k in range(K):
                new_q = initial_q[s, k] + d_sk_n[s, k] - x_sk_n[s, k] + a_sk_n[s, k]
                new_q = max(0.0, new_q)
                
                if new_q > self.config.MAX_QUEUE_STORAGE:
                    total_dropped += (new_q - self.config.MAX_QUEUE_STORAGE)
                    new_q = self.config.MAX_QUEUE_STORAGE
                
                self.queue_lengths[s, k] = new_q
        
        # 5. 综合指标收集
        total_queue_pkts = float(np.sum(self.queue_lengths))
        covered_slots = self.config.NUM_SATELLITES * getattr(self.config, 'NUM_CELLS_PER_SAT', 0)
        if covered_slots > 0:
            avg_q_len = total_queue_pkts / covered_slots
        else:
            avg_q_len = 0.0
        total_arrived = np.sum(a_sk_n)
        current_drop_rate = total_dropped / total_arrived if total_arrived > 0 else 0.0
        
        energy_tx = np.sum(P_matrix) * self.config.TIME_SLOT_DURATION
        energy_isl = 0
        for r in range(S):
            for s in range(S):
                if r != s: # 任何方向上的传输都需要能耗
                    c_rs = np.sum(np.abs(B_tensor[r, s, :]))
                    if c_rs > 0:
                        t_rs = c_rs * self.config.PACKET_SIZE / self.config.ISL_DATA_RATE
                        energy_isl += self.config.ISL_POWER_CONSUMPTION * t_rs
        #total_energy = energy_tx + energy_isl
        total_energy = energy_tx
        
        avg_power = total_energy / self.config.TIME_SLOT_DURATION
        total_throughput = np.sum(x_sk_n)
        
        self.history_metrics['avg_queue'].append(avg_q_len)
        self.history_metrics['avg_power'].append(avg_power)
        self.history_metrics['total_throughput'].append(total_throughput)
        self.history_metrics['drop_rate'].append(current_drop_rate)

        return {
            'avg_queue': avg_q_len,
            'avg_power': avg_power,
            'energy_consumption': total_energy,
            'throughput': total_throughput,
            'drop_rate': current_drop_rate,
            'R_pkts_cap': R_pkts_cap,
            'x_sk_n': x_sk_n
        }