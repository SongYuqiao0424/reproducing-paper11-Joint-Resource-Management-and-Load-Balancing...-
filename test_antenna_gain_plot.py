import os
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from env.channel_model import ChannelModel


def linear_to_db(gain_linear):
    gain_linear = np.maximum(gain_linear, 1e-20)
    return 10.0 * np.log10(gain_linear)


def plot_tx_gain(channel_model, save_dir):
    thetas = np.linspace(0, 180, 1000)
    gains_linear = np.array([channel_model.get_tx_antenna_gain(theta) for theta in thetas])
    gains_db = linear_to_db(gains_linear)

    plt.figure(figsize=(8, 5))
    plt.plot(thetas, gains_db, color='tab:blue', linewidth=2)
    plt.xlabel('Off-Axis Angle (deg)')
    plt.ylabel('Transmit Antenna Gain (dB)')
    plt.title('Transmit Antenna Gain vs Off-Axis Angle')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tx_antenna_gain_1528.png'), dpi=300)


def plot_rx_gain(channel_model, save_dir):
    thetas = np.linspace(0, 180, 1000)
    gains_linear = np.array([channel_model.get_rx_antenna_gain(theta) for theta in thetas])
    gains_db = linear_to_db(gains_linear)

    plt.figure(figsize=(8, 5))
    plt.plot(thetas, gains_db, color='tab:orange', linewidth=2)
    plt.xlabel('Off-Axis Angle (deg)')
    plt.ylabel('Receive Antenna Gain (dB)')
    plt.title('Receive Antenna Gain vs Off-Axis Angle')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rx_antenna_gain_465.png'), dpi=300)


def main():
    config = Config()
    channel_model = ChannelModel(config)

    save_dir = '仿真结果'
    os.makedirs(save_dir, exist_ok=True)

    plot_tx_gain(channel_model, save_dir)
    plot_rx_gain(channel_model, save_dir)

    plt.show()


if __name__ == '__main__':
    main()
