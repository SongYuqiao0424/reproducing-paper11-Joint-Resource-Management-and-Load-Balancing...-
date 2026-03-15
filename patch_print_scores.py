import os

file_path = r'C:\Users\30568\Desktop\研究生\研1\论文学习\波束资源分配\2026.1~2\论文11复现\main.py'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

old_str = '''            # --- 新增：打印本时隙各卫星选择了哪些小区 ---
            allocation_strs = []
            for s in range(config.NUM_SATELLITES):
                selected = np.argsort(F_opt[s, :])[-4:][::-1].tolist()  # 始终取最大的4个
                allocation_strs.append(f"Sat{s}: {selected}")
            print("    Beam Allocation -> " + " | ".join(allocation_strs))'''

new_str = '''            # --- 新增：打印本时隙各卫星选择了哪些小区及其对应的 F_opt 分值 ---
            allocation_strs = []
            for s in range(config.NUM_SATELLITES):
                selected_indices = np.argsort(F_opt[s, :])[-4:][::-1].tolist()
                # 将索引和对应的分值组合成字符串，保留2位小数
                details = [f"C{idx}({F_opt[s, idx]:.2f})" for idx in selected_indices]
                allocation_strs.append(f"Sat{s}: [{', '.join(details)}]")
            print("    Beam Allocation -> " + " | ".join(allocation_strs))'''

if old_str in text:
    text = text.replace(old_str, new_str)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully updated print logic to include scores.")
else:
    print("Target string not found in main.py")
