import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_simulation_results(history_metrics, config):
    '''
    根据论文风格绘制仿真结果指标走势图，如：
    1. 平均队列长度随时间的变化
    2. 平均功率随时间的变化
    '''
    os.makedirs('仿真结果', exist_ok=True)
    
    slots = np.arange(1, len(history_metrics['avg_queue']) + 1)
    
    # 1. 绘制平均队列长度
    plt.figure(figsize=(8, 6))
    plt.plot(slots, history_metrics['avg_queue'], label='Proposed Algo', color='b', linewidth=2)
    plt.xlabel('Time Slots')
    plt.ylabel('Average Queue Length (packets)')
    plt.title('Average Queue Length vs Time Slots')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig('仿真结果/avg_queue_length.png', dpi=300)
    plt.close()

    # 2. 绘制平均功率消耗
    plt.figure(figsize=(8, 6))
    plt.plot(slots, history_metrics['avg_power'], label='Proposed Algo', color='r', linewidth=2)
    plt.xlabel('Time Slots')
    plt.ylabel('Average Power Consumption (W)')
    plt.title('Average Power Consumption vs Time Slots')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig('仿真结果/avg_power.png', dpi=300)
    plt.close()

    # 3. 丢包率走势
    plt.figure(figsize=(8, 6))
    plt.plot(slots, np.array(history_metrics['drop_rate']) * 100, label='Proposed Algo', color='g', linewidth=2)
    plt.xlabel('Time Slots')
    plt.ylabel('Drop Rate (%)')
    plt.title('Packet Drop Rate vs Time Slots')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig('仿真结果/drop_rate.png', dpi=300)
    plt.close()


def plot_beam_power_heatmap(p_history_sat0, config):
    '''
    绘制二维表格的热力图，大小为19*50
    用于表示第一个卫星在前50个时隙中为覆盖范围内19个小区分配的功率情况。
    横坐标: 50个时隙
    纵坐标: 第一个卫星覆盖的19个小区
    颜色深浅: 分配的功率大小
    '''
    # 取出卫星0覆盖的19个小区编号
    cells_sat0 = config.OMEGA_S[0]
    
    # 时隙数限制为最大50
    num_slots = min(50, len(p_history_sat0))
    if num_slots == 0:
        return
        
    heatmap_data = np.zeros((len(cells_sat0), num_slots))
    
    # 填充数据
    for n in range(num_slots):
        P_matrix_n = p_history_sat0[n] # 形如 (L, S, K)
        # 对频段求和，得到该时隙分配给小区的总功率
        for idx, k in enumerate(cells_sat0):
            pow_sum = np.sum(P_matrix_n[:, 0, k])
            heatmap_data[idx, n] = pow_sum
            
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(
        heatmap_data, 
        cmap='YlOrRd', 
        linewidths=.5, 
        linecolor='lightgray', 
        cbar_kws={'label': 'Allocated Power (W)'}
    )
    ax.set_title('Beam Power Allocation Heatmap for Satellite 0')
    ax.set_xlabel('Time Slots (1 to 50)')
    ax.set_ylabel('Cell Index in Coverage Area (1 to 19)')
    
    # 设置刻度标签
    ax.set_yticks(np.arange(len(cells_sat0)) + 0.5)
    ax.set_yticklabels([f"Cell {k}" for k in cells_sat0], rotation=0)
    
    # 横坐标时隙可适当间隔显示
    ax.set_xticks(np.arange(0, num_slots, 5) + 0.5)
    ax.set_xticklabels(np.arange(1, num_slots + 1, 5), rotation=0)
    
    plt.tight_layout()
    plt.savefig('仿真结果/beam_power_heatmap_sat0.png', dpi=300)
    plt.close()
    

def plot_beam_selection_heatmap(f_history_sat0, config):
    '''
    绘制二维表格的热力图，大小为19*50。
    用于表示第一个卫星在前50个时隙中对覆盖范围内19个小区波束的选择情况。
    横坐标: 50个时隙 (前50个)
    纵坐标: 第一个卫星覆盖内的19个小区编号
    选择的小区涂为红色，未选择的为白色。
    '''
    import matplotlib.colors as mcolors
    # 取出卫星0覆盖的19个小区编号
    cells_sat0 = config.OMEGA_S[0]
    
    # 时隙数限制为最大50
    num_slots = min(50, len(f_history_sat0))
    if num_slots == 0:
        return
        
    # 初始化热力图数据矩阵 (19 x num_slots)
    heatmap_data = np.zeros((len(cells_sat0), num_slots))
    
    # 填充热力图数据
    for n in range(num_slots):
        # F_matrix_n 形如 (S, K)，表示该时隙下所有卫星和各小区连接状态
        F_matrix_n = f_history_sat0[n] 
        for idx, k in enumerate(cells_sat0):
            # 将 F_matrix_n 中该小区状态 (1.0 联通 或 0.0 断开) 赋值给绘图数据
            heatmap_data[idx, n] = F_matrix_n[0, k]
            
    # 设置画布大小
    plt.figure(figsize=(14, 8))
    
    # 自定义颜色映射：0.0 对应白色(未选择)，1.0 对应红色(选择)
    cmap = mcolors.ListedColormap(['white', 'red'])
    
    # 绘制热力图并去掉默认的值渐变刻度条 cbar (因为只有0和1)
    ax = sns.heatmap(
        heatmap_data, 
        cmap=cmap, 
        linewidths=.5, 
        linecolor='lightgray', 
        cbar=False
    )
    
    # 设置标题与坐标轴标签
    ax.set_title('Beam Selection Heatmap for Satellite 0')
    ax.set_xlabel('Time Slots (1 to 50)')
    ax.set_ylabel('Cell Index in Coverage Area (1 to 19)')
    
    # 设置Y轴刻度：显示具体的小区编号，旋转0度保证横向易读
    ax.set_yticks(np.arange(len(cells_sat0)) + 0.5)
    ax.set_yticklabels([f"Cell {k}" for k in cells_sat0], rotation=0)
    
    # 设置X轴刻度：显示时隙编号，每5个时隙作为一个刻度以防数字重叠拥挤
    ax.set_xticks(np.arange(0, num_slots, 5) + 0.5)
    ax.set_xticklabels(np.arange(1, num_slots + 1, 5), rotation=0)
    
    # 调整布局缩进并保存图片
    plt.tight_layout()
    plt.savefig('仿真结果/beam_selection_heatmap_sat0.png', dpi=300)
    plt.close()
