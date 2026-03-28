import numpy as np
import scipy.constants as const
import math

class ChannelModel:
    def __init__(self, config):
        self.config = config
        # 从 dBm/Hz 转换为常数瓦特 (W/Hz)
        self.noise_power = 10 ** (self.config.NOISE_PSD_DBW / 10)  # 从 dBW/Hz 转为 线性瓦特
        self.sat_positions, self.cell_positions = self._build_static_geometry()

    def _build_static_geometry(self):
        '''
        构建静态几何拓扑：
        - 地面小区：99个六边形小区中心点，形成紧密蜂窝状
        - 卫星：3条轨道、每轨3颗卫星（忽略轨道倾角）
        '''
        S = self.config.NUM_SATELLITES
        K = self.config.NUM_CELLS

        if S != 9:
            raise ValueError('Current topology builder expects NUM_SATELLITES=9 (3 orbits x 3 satellites).')

        # 1) 生成六边形蜂窝小区中心（轴坐标 -> 笛卡尔坐标）
        # 说明：这里将 CELL_RADIUS 视为正六边形外接圆半径。
        # 点顶式蜂窝布局中心间距：dx = sqrt(3)*R, dy = 1.5*R。
        R = float(self.config.CELL_RADIUS)
        if R <= 0:
            raise ValueError('CELL_RADIUS must be > 0.')

        # 使用轴坐标系生成六边形小区中心点，直到覆盖至少K个小区，ring表示有几环(如1环：中心+6个邻接小区)
        ring = 0
        while 1 + 3 * ring * (ring + 1) < K:
            ring += 1

        # 轴坐标系坐标：q轴：指向"东北"方向（60°角）；r轴：指向"西北"方向（120°角）
        # ( q,  r) 周围邻居：(q+1, r)→东北方向；(q,   r+1)→西北方向；(q-1, r+1)→西方向
        #                   (q-1, r)→西南方向；(q,   r-1)→东南方向；(q+1, r-1)→东方向
        axial_coords = []
        for q in range(-ring, ring + 1):
            r_min = max(-ring, -q - ring)
            r_max = min(ring, -q + ring)
            for r in range(r_min, r_max + 1):
                axial_coords.append((q, r))

        # 选取离中心最近的K个小区，形成紧凑蜂窝簇：距离² = q² + q*r + r²
        # 环数相同时距离相同，故采用多级排序：key=lambda qr: (距离², qr[0], qr[1])
        # key=lambda qr 表示用匿名函数值排序，函数参数qr代表列表中的每个元素
        axial_coords.sort(key=lambda qr: (qr[0] ** 2 + qr[0] * qr[1] + qr[1] ** 2, qr[0], qr[1]))
        axial_coords = axial_coords[:K]

        # 将轴坐标转换为笛卡尔坐标
        cell_positions = np.zeros((K, 3), dtype=float)
        for idx, (q, r) in enumerate(axial_coords):
            x = np.sqrt(3.0) * R * (q + 0.5 * r)
            y = 1.5 * R * r
            cell_positions[idx, 0] = x
            cell_positions[idx, 1] = y
            cell_positions[idx, 2] = 0.0

        # 2) 生成3轨道x3卫星拓扑（忽略轨道倾角，轨道近似平行）
        sat_positions = np.zeros((S, 3), dtype=float)

        cell_extent_x = float(np.max(np.abs(cell_positions[:, 0])))
        cell_extent_y = float(np.max(np.abs(cell_positions[:, 1])))

        cfg_spacing = float(self.config.INTER_SAT_DISTANCE)
        if cfg_spacing <= 0:
            raise ValueError('INTER_SAT_DISTANCE must be > 0.')

        # NOTE: 若配置中的星间距与蜂窝簇尺度不匹配，为保证9星对99小区形成更合理覆盖，
        # 这里对轨道内间距和轨道间距做自适应
        # 先计算理想间距，一个轨道有三个卫星覆盖整体小区的直径，则间距应为整体小区2/3半径。
        ideal_in_orbit = max(0.66*cell_extent_x, 1.0)
        ideal_cross_orbit = max(0.66*cell_extent_y, 1.0)

        # 计算实际轨道内间距和轨道间距，若与理想间距不符，则进行适当调整
        in_orbit_spacing = float(np.clip(cfg_spacing, 0.7 * ideal_in_orbit, 1.3 * ideal_in_orbit))
        cross_orbit_spacing = float(np.clip(cfg_spacing, 0.7 * ideal_cross_orbit, 1.3 * ideal_cross_orbit))

        x_offsets = np.array([-in_orbit_spacing, 0.0, in_orbit_spacing], dtype=float)
        y_orbits = np.array([-cross_orbit_spacing, 0.0, cross_orbit_spacing], dtype=float)

        # NOTE: 静态投影简化模型中，三条轨道在x-y平面统一左旋30°（逆时针）。
        # 若需与配置联动，可将30°替换为配置参数；当前不使用 config.INCLINATION。
        orbit_tilt_deg = 30.0
        orbit_tilt_rad = np.radians(orbit_tilt_deg)
        rot_mat = np.array([
            [np.cos(orbit_tilt_rad), -np.sin(orbit_tilt_rad)],
            [np.sin(orbit_tilt_rad),  np.cos(orbit_tilt_rad)],
        ])

        sat_idx = 0
        for y_orbit in y_orbits:
            for x_off in x_offsets:
                xy_rot = rot_mat @ np.array([x_off, y_orbit], dtype=float)
                sat_positions[sat_idx, 0] = xy_rot[0]
                sat_positions[sat_idx, 1] = xy_rot[1]
                sat_positions[sat_idx, 2] = self.config.ORBIT_ALTITUDE
                sat_idx += 1

        return sat_positions, cell_positions

    @staticmethod
    def _angle_between(v1, v2):
        '''计算两个向量夹角(度)。'''
        # 计算向量欧几里得长度
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 <= 0 or norm2 <= 0:
            return 0.0
        # 点积运算求解夹角余弦值，并转换为角度
        cos_val = np.dot(v1, v2) / (norm1 * norm2)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))

    def calculate_free_space_path_loss(self, distance, freq):
        '''
        计算自由空间路径损耗 (FSPL) - 线性值
        FSPL(dB) = 20 * log10(d) + 20 * log10(f) + 20 * log10(4 * pi / c)
        '''
        lambda_c = const.c / freq
        fspl_linear = ( (4 * math.pi * distance) / lambda_c ) ** 2
        return fspl_linear

    def get_tx_antenna_gain(self, theta, theta_3db=2.5, G_max=32.44, L_n=-20.0):
        '''
        发射端天线增益 (ITU-R S.1528 分段模型)
        返回线性增益
        :param theta: 偏离波束中心的角度(度)
        :param theta_3db: Phi0，3dB波束宽度的一半(度)
        :param G_max: 最大天线增益 dB 
        :param L_n: 旁瓣电平(dB)
        '''
        if theta < 0 or theta > 180:
            raise ValueError('Invalid value for off-axis angle theta, must be in [0, 180].')
        if G_max <= 0:
            raise ValueError('Invalid value for G_max, must be > 0.')
        if theta_3db <= 0:
            raise ValueError('Invalid value for theta_3db(Phi0), must be > 0.')

        a = 2.58
        b = 6.32
        alpha = 1.5

        x_term = G_max + L_n + 25 * np.log10(b * theta_3db)
        y_term = (10 ** (0.04 * (G_max + L_n))) * b * theta_3db

        if theta <= a * theta_3db:
            gain_db = G_max - 3 * (theta / theta_3db) ** alpha
        elif theta <= b * theta_3db:
            gain_db = G_max + L_n
        elif theta <= y_term:
            gain_db = x_term - 25 * np.log10(theta)
        elif theta <= 90:
            gain_db = 0.0
        else:
            gain_db = max(15 + L_n + 0.25 * G_max, 0.0)
        
        return 10 ** (gain_db / 10)

    def get_rx_antenna_gain(self, theta, G_max=26.42):
        '''
        接收端天线增益 (ITU-R S.465 分段模型)
        返回线性增益
        '''
        if theta < 0 or theta > 180:
            raise ValueError('Invalid value for off-axis angle theta, must be in [0, 180].')
        if G_max <= 0:
            raise ValueError('Invalid value for G_max, must be > 0.')

        d_lambda = np.sqrt((10 ** (G_max / 10)) / (0.7 * np.pi ** 2))
        phi_b = 10 ** (42 / 25)

        if d_lambda > 100:
            g1 = 32.0
            phi_r = 1.0
        else:
            g1 = -18 + 25 * np.log10(d_lambda)
            phi_r = 100 / d_lambda

        phi_m = 20 / d_lambda * np.sqrt(G_max - g1)

        if phi_b < phi_r:
            raise ValueError('Invalid S.465 parameters: phi_b is less than phi_r.')
        if G_max < g1:
            raise ValueError('Invalid S.465 parameters: G_max is less than G1.')

        if theta <= phi_m:
            gain_db = G_max - 2.5e-3 * (d_lambda * theta) ** 2
        elif theta <= phi_r:
            gain_db = g1
        elif theta <= phi_b:
            gain_db = 32 - 25 * np.log10(theta)
        else:
            gain_db = -10.0
            
        return 10 ** (gain_db / 10)

    def compute_channel_coefficients(self, distance, tx_theta, rx_theta_leo, rx_theta_gso):
        '''
        distance: 卫星与地面小区之间的距离 (m)
        tx_theta: 发射天线偏离波束中心的角度 (度)
        rx_theta_leo: LEO用户接收端偏离角 (度)
        rx_theta_gso: GSO接收端偏离角 (度)
        同时计算信道系数幅度 |h| 与 |g|:
        |h| = sqrt((G_tx * G_rx_leo) / P_loss)
        |g| = sqrt((G_tx * G_rx_gso) / P_loss)
        '''
        # 计算自由空间路径损耗 (线性)
        path_loss_linear = self.calculate_free_space_path_loss(distance, self.config.CARRIER_FREQ)
        
        # 计算天线增益 (线性)
        g_tx = self.get_tx_antenna_gain(tx_theta)
        g_rx_leo = self.get_rx_antenna_gain(rx_theta_leo)
        g_rx_gso = self.get_rx_antenna_gain(rx_theta_gso)
        
        # 信道系数功率
        h_squared = (g_tx * g_rx_leo) / path_loss_linear
        g_squared = (g_tx * g_rx_gso) / path_loss_linear
        
        # 返回幅度
        return math.sqrt(max(h_squared, 0.0)), math.sqrt(max(g_squared, 0.0))

    def compute_channel_coefficient(self, distance, tx_theta, rx_theta):
        '''
        向后兼容接口：仅返回 |h|。
        '''
        h_amp, _ = self.compute_channel_coefficients(distance, tx_theta, rx_theta, rx_theta)
        return h_amp

    def generate_random_channel_matrices(self):
        '''
        最小接入版几何信道生成：
        - 使用轨道高度、倾角、星间距、小区半径构建静态拓扑
        - 根据几何关系计算发射端偏离角、接收端偏离角与传播距离
        - 同时生成 LEO 信道 H 与 GSO 干扰信道 G
        针对公式(1) 与公式(4) 
        返回: 
        H: [|S|, |K|, |K|] -> h_{s,k,j}, 卫星s指向j的波束在k处接收到的信道增益
        G: [|S|, |K|, |K|] -> g_{s,k,j}, 卫星s指向j的波束对GSO地面站k处的干扰信道增益
        '''
        S = self.config.NUM_SATELLITES
        K = self.config.NUM_CELLS
        gso_cells = set(getattr(self.config, 'K_G', []))
        # 假设地面小区位于平坦地面，小区接收天线方向为z轴正向
        local_zenith = np.array([0.0, 0.0, 1.0])
        
        H = np.zeros((S, K, K))
        G_matrix = np.zeros((S, K, K))
        
        for s in range(S):
            # 使用向量计算对于卫星s，各小区位置的向量
            sat_pos = self.sat_positions[s]
            sat_to_cells = self.cell_positions - sat_pos
            distances = np.linalg.norm(sat_to_cells, axis=1)

            # 小区外层循环j表示卫星s指向的小区，内层循环k表示接收端小区
            # 即此时计算小区k接收来自卫星s指向小区j的信号强度
            for j in range(K):
                # 卫星s指向小区j的波束主轴向量
                beam_boresight_vec = sat_to_cells[j]
                for k in range(K):
                    # 卫星s指向小区k的信号向量
                    rx_vec = sat_to_cells[k]

                    tx_theta = self._angle_between(beam_boresight_vec, rx_vec)

                    # 小区k指向卫星s的向量
                    cell_to_sat_vec = sat_pos - self.cell_positions[k]
                    rx_theta_leo = self._angle_between(cell_to_sat_vec, local_zenith)
                    rx_theta_gso = rx_theta_leo

                    # 约束：当 LEO 波束指向小区k且该小区存在 GSO 接收端时，
                    # GSO 偏离角至少比对应 LEO 偏离角大 10°
                    if j == k and k in gso_cells:
                        rx_theta_gso = min(180.0, max(rx_theta_gso, rx_theta_leo + 10.0))

                    h_amp, g_amp = self.compute_channel_coefficients(
                        distance=distances[k],
                        tx_theta=tx_theta,
                        rx_theta_leo=rx_theta_leo,
                        rx_theta_gso=rx_theta_gso,
                    )

                    # 轻微随机扰动，保留随时隙轻变特性
                    H[s, k, j] = h_amp * np.random.uniform(0.95, 1.05)
                    G_matrix[s, k, j] = g_amp * np.random.uniform(0.95, 1.05)
                    
        return H, G_matrix

