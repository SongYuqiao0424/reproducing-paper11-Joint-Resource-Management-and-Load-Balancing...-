
with open('main.py', 'r', encoding='utf-8') as f:
    text = f.read()

bad = '''        # 屏幕显示进度
        if (n + 1) % 100 == 0:
            print(f'[Slot {n+1:4d} / {config.MAX_TIME_SLOTS}] Avg Queue: {step_metrics[\
avg_queue\]:.2f} pkts | EnergyTx: {step_metrics[\energy_consumption\]:.4f} J | Tput: {step_metrics[\throughput\]:.2f}')'''

good = '''        # 屏幕显示进度
        if (n + 1) % 100 == 0:
            avg_q = step_metrics['avg_queue']
            erg = step_metrics['energy_consumption']
            tpt = step_metrics['throughput']
            print(f'[Slot {n+1:4d} / {config.MAX_TIME_SLOTS}] Avg Queue: {avg_q:.2f} pkts | EnergyTx: {erg:.4f} J | Tput: {tpt:.2f}')'''

text = text.replace(bad, good)
with open('main.py', 'w', encoding='utf-8') as f:
    f.write(text)

