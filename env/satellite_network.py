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
            'total_throughput': []
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
        
        # 将 Mbps 转化为这个时隙内的数据包个数 (以 1Mbits 为 1 包)
        # Mbps -> 10^6 bits / s. 
        # lambda_pkts = (Rate(bits/s) * T0) / M0
        lambda_list = (self.current_arrival_rates * self.config.TIME_SLOT_DURATION) / self.config.PACKET_SIZE
        
        # 对每个小区生成泊松分布的到达包数
        arrived_pkts_for_cells = np.random.poisson(lambda_list)
        
        # 构建并分配到所属服务卫星，分配策略：平均分配排队
        a_sk_n = np.zeros((S, K))
        for k in range(K):
            # 简单的均匀分发请求至所有覆盖该小区的卫星 (根据论文 Φ(k) 关联)
            pkts_per_sat = arrived_pkts_for_cells[k] // S
            remainder = arrived_pkts_for_cells[k] % S
            
            for s in range(S):
                a_sk_n[s, k] = pkts_per_sat
                if s < remainder: # 分配余数
                    a_sk_n[s, k] += 1
                    
        return a_sk_n

    def update_queues(self, received_a, routed_d, transmitted_x):
        '''
        根据论文公式(8)更新队列
        q_{s,k}[n+1] = q_{s,k}[n] + d_{s,k}[n] + a_{s,k}[n] - x_{s,k}[n]
        '''
        S = self.config.NUM_SATELLITES
        K = self.config.NUM_CELLS
        
        for s in range(S):
            for k in range(K):
                # 更新队列模型，且保证队列非负以及不超过最高容量
                new_q = self.queue_lengths[s, k] + routed_d[s, k] + received_a[s, k] - transmitted_x[s, k]
                new_q = max(0.0, new_q)
                
                # 丢包限制 (如果有最大存储容量)
                if new_q > self.config.MAX_QUEUE_STORAGE:
                    new_q = self.config.MAX_QUEUE_STORAGE
                
                self.queue_lengths[s, k] = new_q

    def step(self, F_pattern, P_matrix, B_tensor):
        '''
        通过外部 BCD 算法求解到的 F, P, B 动作执行状态步进。
        计算信噪比 -> 可达速率 R -> 实际传输包数 x -> 更新队列。
        返回本时隙的综合性能指标字典。
        '''
        # 1. 漂移与到达
        self.current_time_slot += 1
        if self.current_time_slot % self.config.DEMAND_DRIFT_STEPS == 0:
            self._drift_arrival_rates()
            
        a_sk_n = self.generate_arrivals()
        
        # 2. 从星际链路矩阵转移量 B 中计算队列变动 d_{s,k}
        S = self.config.NUM_SATELLITES
        K = self.config.NUM_CELLS
        d_sk_n = np.zeros((S, K))
        for s in range(S):
            for k in range(K):
                # 汇总其他所有卫星 r 转移给 s 的属于目标小区 k 的包
                d_sk_n[s, k] = np.sum(B_tensor[:, s, k]) 
        
        # 3. 计算实时发射速率与有效发包数目 x_{s,k} 
        x_sk_n = np.zeros((S, K))
        
        # 这里为了简化框架演示，跳过了实际复杂的香农信干噪比算子，
        # 在完整主干接入后，这一步需要在外部或者内部将 P 转换为香农速率 R_sk.
        # 我们假设外部优化或此步骤已知每个链路给出的实际可下发数据包容量 R_pkts
        # R_pkts = np.random.uniform(0, 100, size=(S,K)) * F_pattern ...
        R_pkts = F_pattern * (np.sum(P_matrix, axis=0) * 0.5) # 模拟占位符评估
        
        for s in range(S):
            for k in range(K):
                # 公式 (7)
                available_to_send = self.queue_lengths[s, k] + d_sk_n[s, k]
                x_sk_n[s, k] = min(R_pkts[s, k], available_to_send)
                x_sk_n[s, k] = max(0, x_sk_n[s, k])
        
        # 4. 更新物理队列
        self.update_queues(a_sk_n, d_sk_n, x_sk_n)
        
        # 5. 记录指标等...
        avg_q_len = np.mean(self.queue_lengths)
        energy_tx = np.sum(P_matrix) * self.config.TIME_SLOT_DURATION
        # (这里未累加ISL总开销作简化展示)
        
        return {
            'avg_queue': avg_q_len,
            'energy_consumption': energy_tx,
            'throughput': np.sum(x_sk_n)
        }

