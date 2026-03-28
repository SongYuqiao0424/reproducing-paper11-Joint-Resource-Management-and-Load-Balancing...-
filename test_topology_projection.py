import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

from config import Config
from env.channel_model import ChannelModel


def plot_topology_projection(config, channel_model):
    cell_pos = channel_model.cell_positions
    sat_pos = channel_model.sat_positions

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制地面六边形蜂窝（仅投影到x-y平面）
    # 按轴坐标邻接方向约定，将当前六边形朝向整体旋转90°。
    hex_orientation = np.pi / 6.0 + np.pi / 2.0
    for idx, (x, y, _) in enumerate(cell_pos):
        hex_patch = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=config.CELL_RADIUS,
            orientation=hex_orientation,
            facecolor='none',
            edgecolor='lightgray',
            linewidth=0.8,
            alpha=0.9,
        )
        ax.add_patch(hex_patch)

    # 小区中心点
    ax.scatter(cell_pos[:, 0], cell_pos[:, 1], s=12, c='royalblue', alpha=0.8, label='Cell Centers')

    # 绘制三条轨道（每轨3星）：按当前拓扑生成顺序切分 [0:3], [3:6], [6:9]
    # NOTE: 该切分假设 channel_model._build_static_geometry 中采用 y_orbits 外层循环、x_offsets 内层循环。
    orbit_indices = [list(range(0, 3)), list(range(3, 6)), list(range(6, 9))]
    orbit_colors = ['tab:red', 'tab:green', 'tab:purple']

    for orbit_id, (indices, color) in enumerate(zip(orbit_indices, orbit_colors), start=1):
        orbit_xy = sat_pos[indices, :2]
        order = np.argsort(orbit_xy[:, 0])
        orbit_xy = orbit_xy[order]

        ax.plot(
            orbit_xy[:, 0],
            orbit_xy[:, 1],
            linestyle='--',
            linewidth=2.0,
            color=color,
            alpha=0.85,
            label=f'Orbit {orbit_id}',
        )

    # 卫星投影点
    ax.scatter(
        sat_pos[:, 0],
        sat_pos[:, 1],
        s=180,
        marker='*',
        c='gold',
        edgecolors='black',
        linewidths=0.9,
        zorder=5,
        label='Satellites (Projected)',
    )

    # 标注卫星索引
    for s_idx, (sx, sy, _) in enumerate(sat_pos):
        ax.text(sx, sy + 0.08 * config.CELL_RADIUS, f'S{s_idx}', fontsize=9, ha='center', va='bottom')

    ax.set_title('Ground Honeycomb + 3-Orbit 9-Satellite Projection')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.35)

    # 统一图例去重
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='upper right')

    plt.tight_layout()

    os.makedirs('仿真结果', exist_ok=True)
    save_path = os.path.join('仿真结果', 'topology_projection_honeycomb_3orbits_9sats.png')
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    cfg = Config()
    ch_model = ChannelModel(cfg)
    plot_topology_projection(cfg, ch_model)
