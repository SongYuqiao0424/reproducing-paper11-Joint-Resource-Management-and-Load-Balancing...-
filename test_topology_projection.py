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
    cell_patches = []

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
        cell_patches.append(hex_patch)

    # 小区中心点
    ax.scatter(cell_pos[:, 0], cell_pos[:, 1], s=12, c='royalblue', alpha=0.8, label='Cell Centers')

    # 标注小区编号（与拓扑实际编号一致）
    label_y_offset = 0.10 * config.CELL_RADIUS
    for c_idx, (cx, cy, _) in enumerate(cell_pos):
        ax.text(
            cx,
            cy + label_y_offset,
            f'C{c_idx}',
            fontsize=6,
            color='dimgray',
            alpha=0.85,
            ha='center',
            va='center',
            zorder=4,
        )

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

    # 卫星投影点（开启 picker 以支持点击交互）
    sat_scatter = ax.scatter(
        sat_pos[:, 0],
        sat_pos[:, 1],
        s=180,
        marker='*',
        c='gold',
        edgecolors='black',
        linewidths=0.9,
        zorder=5,
        picker=True,
        pickradius=8,
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

    # 点击卫星后高亮其覆盖小区；再次点击同一卫星恢复无色。
    selected_sat = {'idx': None}
    default_sat_color = np.array([1.0, 0.843, 0.0, 1.0])
    active_sat_color = np.array([1.0, 0.35, 0.1, 1.0])

    def refresh_highlight():
        # 初始化卫星颜色
        sat_colors = np.tile(default_sat_color, (config.NUM_SATELLITES, 1))

        for k, patch in enumerate(cell_patches):
            # 若当前选中卫星覆盖当前小区，则高亮显示；否则保持默认无色。
            if selected_sat['idx'] is not None and k in config.OMEGA_S.get(selected_sat['idx'], []):
                patch.set_facecolor('#ffcf40')
                patch.set_edgecolor('#b8860b')
                patch.set_alpha(0.55)
            else:
                patch.set_facecolor('none')
                patch.set_edgecolor('lightgray')
                patch.set_alpha(0.9)

        # 更新卫星颜色和图像标题
        if selected_sat['idx'] is not None:
            sat_colors[selected_sat['idx']] = active_sat_color
            covered_num = len(config.OMEGA_S.get(selected_sat['idx'], []))
            ax.set_title(
                f'Ground Honeycomb + 3-Orbit 9-Satellite Projection | '
                f'S{selected_sat["idx"]} covers {covered_num} cells'
            )
        else:
            ax.set_title('Ground Honeycomb + 3-Orbit 9-Satellite Projection')

        # 更新配置并刷新图像
        sat_scatter.set_facecolors(sat_colors)
        fig.canvas.draw_idle()

    def on_pick(event):
        if event.artist != sat_scatter or len(event.ind) == 0:
            return

        sat_idx = int(event.ind[0])
        # 切换选择状态：点击同一卫星取消选择，点击不同卫星选择新卫星
        if selected_sat['idx'] == sat_idx:
            selected_sat['idx'] = None
        else:
            selected_sat['idx'] = sat_idx
        refresh_highlight()

    # 注册鼠标点击事件
    fig.canvas.mpl_connect('pick_event', on_pick)
    refresh_highlight()

    plt.tight_layout()

    os.makedirs('仿真结果', exist_ok=True)
    save_path = os.path.join('仿真结果', 'topology_projection_honeycomb_3orbits_9sats.png')
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    cfg = Config()
    ch_model = ChannelModel(cfg)
    plot_topology_projection(cfg, ch_model)
