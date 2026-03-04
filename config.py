# 仿真参数全局配置
class Config:
    # ---------------- 卫星与网络拓扑参数 (基于 Table II) ----------------
    NUM_SATELLITES = 2            # 卫星数量 |S| (小型场景为2)
    NUM_BEAMS_PER_SAT = 2         # 每颗卫星波束数量 Nb 
    NUM_CELLS = 12                # 目标服务地面小区数量 |K|
    NUM_FREQUENCY_SEGMENTS = 4    # 频率段块数量 |L|
    BANDWIDTH_PER_SEGMENT = 100e6 # 每个频率段带宽 W (100 MHz)
    CARRIER_FREQ = 20e9           # 载波频率 (20 GHz, Ka-band)
    
    # 卫星运动分布属性
    ORBIT_ALTITUDE = 1000e3       # LEO 卫星轨道高度 (1000 km)
    INCLINATION = 85.0            # 轨道倾角 (85 弧度)
    INTER_SAT_DISTANCE = 207e3    # 相邻 LEO 星间距离 (207 km)

    # ---------------- 发射与接收硬件参数 ----------------
    MAX_POWER_PER_SAT = 100.0     # 每颗卫星最大总允许功率上下限 Pmax (单位: Watts)
    NOISE_PSD_DBM = -174          # 加性高斯白噪声谱密度 N0 (dBm/Hz)

    # ---------------- 业务需求流模型 (排队与负载) ----------------
    TIME_SLOT_DURATION = 10e-3    # 时隙时长 T0 (10 ms)
    MAX_TIME_SLOTS = 1500         # 仿真总时隙步数 (即 1500 * 10ms = 15s)
    PACKET_SIZE = 1e6             # 单元数据包大小 M0 (1 Mbits)
    
    # 泊松分布包到达率范围
    ARRIVAL_RATE_MIN = 80e6       # 小区最小包到达均率 (80 Mbps)
    ARRIVAL_RATE_MAX = 320e6      # 小区最大包到达均率 (320 Mbps)
    DEMAND_DRIFT_STEPS = 50       # 平均到达率随机漂移间隔时隙 (每 50 slots漂移更新一次)
    
    MAX_QUEUE_STORAGE = 350       # 卫星节点允许对该小区排队的最大容量 (Mbits)

    # ---------------- 卫星与小区覆盖关系集合 ----------------
    OMEGA_S = {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [4, 5, 6, 7, 8, 9, 10, 11]}
    PHI_K = {0:[0], 1:[0], 2:[0], 3:[0], 4:[0,1], 5:[0,1], 6:[0,1], 7:[0,1], 8:[1], 9:[1], 10:[1], 11:[1]}

    # ---------------- 星际激光链路 (ISL) 参数 ----------------
    ISL_MAX_TRANSFER_PKTS = 2000  # ISL能承担最大负载转移包数限 c_max (通过 Table II 获得)
    ISL_DATA_RATE = 100e9         # ISL 允许最大有效带宽 R_I (100 Gbps，取值典型激光端机 CONDOR)
    ISL_POWER_CONSUMPTION = 50.0  # 单个 ISL 发射机设备功耗 P_I (Watts)

    # ---------------- GSO 地面站共存避扰参数 ----------------
    K_G = [2, 8]  # 存在 GSO 地面站的小区集合 k
    L_K = {2: [0, 1], 8: [2, 3]}  # 受到重叠干扰验证的频段集合
    Z_MAX_DBW = -140              # 允许对 GSO 端最大干扰门限 Zmax (dBW 级别, 根据Fig 4设定)

    # ---------------- 算法与李雅普诺夫权衡控制参数 ----------------
    V = 200                       # V 控制权衡参数 (Figure 3 选择 200 折中延迟与功耗)
    L_0 = 1e6                     # 队列长度规范项 L_0
    E_0 = 1.0                     # 能效缩放项 E_0
    
    # 优化求解迭代终止次数限制
    MAX_BCD_LOOPS = 5             # 外层坐标块下降控制最大迭代数 (Fig 2 表明 4~5轮即收敛)
    MPMM_THETA_ROUNDS = 10        # 乘子惩罚法 (MP) 外层加强循环轮数 \Theta
    MPMM_PSI_ROUNDS = 5           # 优化最小 (MM) 内部迭代逼近 \Psi 数
    SCA_XI_ROUNDS = 5             # 连续凸逼近 (SCA) 阶段更新数 \Xi

