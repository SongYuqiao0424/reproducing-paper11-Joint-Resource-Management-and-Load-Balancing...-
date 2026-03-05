import numpy as np
import scipy.constants as const
import math

class ChannelModel:
    def __init__(self, config):
        self.config = config
        # 从 dBm/Hz 转换为常数瓦特 (W/Hz)
        self.noise_power = (10 ** (self.config.NOISE_PSD_DBM / 10)) * 1e-3

    def calculate_free_space_path_loss(self, distance, freq):
        '''
        计算自由空间路径损耗 (FSPL) - 线性值
        FSPL(dB) = 20 * log10(d) + 20 * log10(f) + 20 * log10(4 * pi / c)
        '''
        lambda_c = const.c / freq
        fspl_linear = ( (4 * math.pi * distance) / lambda_c ) ** 2
        return fspl_linear

    def get_tx_antenna_gain(self, theta, theta_3db=2.5, G_max=40.0):
        '''
        发射端天线增益 (参考 ITU-R S.1528)
        这里简化为一个标准的降级函数, 返回线性增益
        :param theta: 偏离波束中心的角度(度)
        :param theta_3db: 3dB波束宽度，通常 Ka 频段很窄
        :param G_max: 最大天线增益 dB 
        '''
        # 抛物线天线辐射方向图近似
        u = 2.07123 * np.sin(np.radians(theta)) / np.sin(np.radians(theta_3db))
        if theta == 0:
            gain_db = G_max
        else:
            # 近似第一旁瓣电平
            J1 = np.abs(2 * const.pi * u)
            if J1 == 0:
                gain_db = G_max
            else:
                gain_db = G_max - 12 * (theta / theta_3db)**2
                # 对最小增益进行截断
                gain_db = max(gain_db, -20.0) 
        
        return 10 ** (gain_db / 10)

    def get_rx_antenna_gain(self, theta, G_max=35.0):
        '''
        接收端天线增益 (参考 ITU-R S.465-6)
        返回线性增益
        '''
        # GSO 地面站或者用户终端天线增益近似
        # G(theta) = 32 - 25 * log10(theta)
        if theta < 1.0:
            gain_db = G_max
        elif theta < 48:
            gain_db = 32 - 25 * np.log10(theta)
        else:
            gain_db = -10.0
            
        return 10 ** (gain_db / 10)

    def compute_channel_coefficient(self, distance, tx_theta, rx_theta):
        '''
        distance: 卫星与地面小区之间的距离 (m)
        tx_theta: 发射天线偏离波束中心的角度 (度)
        rx_theta: 接收天线偏离波束中心的角度 (度)
        计算信道系数 |h|^2 
        h^2 = (G_tx * G_rx) / P_loss
        '''
        # 计算自由空间路径损耗 (线性)
        path_loss_linear = self.calculate_free_space_path_loss(distance, self.config.CARRIER_FREQ)
        
        # 计算天线增益 (线性)
        g_tx = self.get_tx_antenna_gain(tx_theta)
        g_rx = self.get_rx_antenna_gain(rx_theta)
        
        # 信道系数平方
        h_squared = (g_tx * g_rx) / path_loss_linear
        return h_squared

    def generate_random_channel_matrices(self):
        '''
        在缺乏具体经纬度卫星运动轨迹计算时，生成一个仿真的信道矩阵(马尔可夫演变或者静态分布)
        针对公式(1) 与公式(4) 
        返回: 
        H: [|S|, |K|, |K|] -> h_{s,k,j}, 卫星s指向j的波束在k处接收到的信道增益
        G: [|S|, |K|, |K|] -> g_{s,k,j}, 卫星s指向j的波束对GSO地面站k处的干扰信道增益
        '''
        S = self.config.NUM_SATELLITES
        K = self.config.NUM_CELLS
        
        
        # 假设高度导致的基线损耗在 -150dB 左右 (10^(-15))，波束对准时增益高一点，不对准衰减大
        # 这是一个随机近似函数供算法测试使用，后续可以换为真实的轨道追踪计算
        base_h = 1e-15 
        base_g = 1e-16  # GSO站受到的底噪略低
        
        H = np.zeros((S, K, K))
        G_matrix = np.zeros((S, K, K))
        
        for s in range(S):
            for k in range(K):
                for j in range(K):
                    if k == j:
                        # 指向当前小区 k，增益最大
                        H[s, k, j] = base_h * np.random.uniform(0.8, 1.2)
                    else:
                        # 指向其他小区 j 对 k 造成的相邻波束干扰
                        dist_factor = np.random.uniform(0.01, 0.1) # 旁瓣衰减
                        H[s, k, j] = base_h * dist_factor
        
        for s in range(S):
            for k in range(K):
                for j in range(K):
                    if k == j:
                        # 目标波束指向 j 与 GSO 所在小区 k 一致，GSO受到主瓣干扰，增益较大
                        G_matrix[s, k, j] = base_g * np.random.uniform(0.8, 1.2)
                    else:
                        # 指向其他小区 j 时，对 k 处的 GSO 造成旁瓣干扰，增益较小
                        dist_factor = np.random.uniform(0.01, 0.1)
                        G_matrix[s, k, j] = base_g * dist_factor
                    
        return H, G_matrix

