# 仿真参数全局配置
class Config:
    # ---------------- 卫星与网络拓扑参数 (基于 Table II) ----------------
    NUM_SATELLITES = 9            # 卫星数量 |S| (9)
    NUM_BEAMS_PER_SAT = 4         # 每颗卫星波束数量 Nb (4)
    NUM_CELLS = 99                # 目标服务地面小区数量 |K| (99)
    NUM_FREQUENCY_SEGMENTS = 4    # 频率段块数量 |L| (4)
    BANDWIDTH_PER_SEGMENT = 50e6  # 每个频率段带宽 W (50 MHz)
    CARRIER_FREQ = 20e9           # 载波频率 (20 GHz, Ka-band)
    
    # 卫星运动分布属性
    ORBIT_ALTITUDE = 1000e3       # LEO 卫星轨道高度 (1000 km)
    INCLINATION = 85.0            # 轨道倾角 (85 弧度，暂未使用，仿真绘图轨道倾角为30°)
    INTER_SAT_DISTANCE = 207e3    # 相邻 LEO 星间距离 (207 km)
    CELL_RADIUS = 39e3            # 小区半径 (39 km)

    # ---------------- 发射与接收硬件参数 ----------------
    MAX_POWER_PER_SAT = 100.0     # 每颗卫星最大总允许功率上下限 Pmax (单位: Watts, 维持原设)
    NOISE_PSD_DBW = -204          # 加性高斯白噪声谱密度 N0 (-204 dBW/Hz)

    # ---------------- 业务需求流模型 (排队与负载) ----------------
    TIME_SLOT_DURATION = 10e-3    # 时隙时长 (10 ms)
    MAX_TIME_SLOTS = 100          # 仿真总时隙步数
    PACKET_SIZE = 5e3             # 单元数据包大小 M0 (5 Kbits)
    
    # 泊松分布包到达率范围
    ARRIVAL_RATE_MIN = 80e6       
    ARRIVAL_RATE_MAX = 250e6      
    DEMAND_DRIFT_STEPS = 50       
    MAX_QUEUE_STORAGE = 10000       

    # ---------------- 卫星与小区覆盖关系集合 ----------------
    # 依照表格：|S|=9, 每个卫星覆盖19个小区，总共99个小区。生成合理的分布覆盖映射：
    OMEGA_S = {s: [(s * 11 + i) % 99 for i in range(19)] for s in range(9)}
    PHI_K = {k: [s for s in range(9) if k in [(s * 11 + i) % 99 for i in range(19)]] for k in range(99)}

    # ---------------- 星际激光链路 (ISL) 参数 ----------------
    ISL_MAX_TRANSFER_PKTS = 2000  
    ISL_DATA_RATE = 10e9          # ISL 允许最大有效带宽 R_I (10 Gbps)
    ISL_POWER_CONSUMPTION = 4.0   # 单个 ISL 发射机设备功耗 P_I (4 W)

    # ---------------- GSO 地面站共存避扰参数 ----------------
    K_G = [10, 50, 80]            # 存在 GSO 地面站的小区集合 k (适当拓展)
    L_K = {10: [0, 1], 50: [2, 3], 80: [0, 3]}  # 受扰GSO小区K的受扰频段
    Z_MAX_DBW = -130              # 允许对 GSO 端最大干扰门限 Zth (-130 dBW)

    # ---------------- 算法与李雅普诺夫权衡控制参数 ----------------、
    # 优化MPMM算法：经过测试发现L_0过大导致优化目标第一项权重过低，引起数值不稳定和solver不精确，适当降低L_0以确保数值稳定性; 
    # L_0过小导致优化目标第一项权重过高，忽略J_mp的优化，陷入局部最优解（0.5陷阱）            
    # 最终参数 ：L_0 = 1e3 ；MPMM_THETA_ROUNDS = 10；MPMM_PSI_ROUNDS = 1； max_iters=10000, eps=1e-4

    # 优化SCA算法：测试发现L_0过大导致功率分配会较为极端，经常将全部功率分配给1/2个波束；
    # V的数值过小时，优化目标对功耗不敏感，导致功率分配过量（即分配功率远大于队列需要功率）
    # 最终参数 ：V = 1000；L_0 = 1e3；E_0 = 6.25；SCA_XI_ROUNDS = 1； max_iters=50000, eps=1e-4

    # 优化QP算法：未测试参数对算法性能的影响，此参数下负载均衡功能正常
    # 最终参数 ：V = 1000；L_0 = 1e3；E_0 = 6.25；
    
    V = 1000                       # 李雅普诺夫权衡控制参数 V (适当调整以平衡能效与队列长度)
    L_0 = 1e3                     # 队列长度规范项 L_0
    E_0 = 6.25                    # 能效缩放项 E_0 (6.25 J)
    
    MAX_BCD_LOOPS = 5             # BCD 主循环最大迭代次数
    MPMM_THETA_ROUNDS = 10        # MPMM 外循环迭代次数
    MPMM_PSI_ROUNDS = 1           # MPMM 内循环迭代次数
    SCA_XI_ROUNDS = 1             # SCA 迭代次数
