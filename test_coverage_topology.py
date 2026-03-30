from collections import Counter

from config import Config
from env.channel_model import ChannelModel


def main():
    cfg = Config()
    _ = ChannelModel(cfg)  # 初始化时会按几何拓扑重建 OMEGA_S / PHI_K

    print('=== Coverage Report: Satellite -> Cell Indices ===')
    for s in range(cfg.NUM_SATELLITES):
        covered_cells = sorted(cfg.OMEGA_S.get(s, []))
        print(f'Sat{s:02d} ({len(covered_cells)} cells): {covered_cells}')

    print('\n=== Coverage Count Distribution: Cells -> #Covering Satellites ===')
    cover_counts = {k: len(cfg.PHI_K.get(k, [])) for k in range(cfg.NUM_CELLS)}
    distribution = Counter(cover_counts.values())

    for num_sats in sorted(distribution.keys()):
        num_cells = distribution[num_sats]
        print(f'{num_sats} satellite(s): {num_cells} cell(s)')

    uncovered_cells = sorted([k for k, c in cover_counts.items() if c == 0])

    print('\n=== Coverage Summary ===')
    if uncovered_cells:
        print(f'Uncovered cells found: {len(uncovered_cells)}')
        print(f'Uncovered cell indices: {uncovered_cells}')
    else:
        print('All cells are covered by at least one satellite.')


if __name__ == '__main__':
    main()
