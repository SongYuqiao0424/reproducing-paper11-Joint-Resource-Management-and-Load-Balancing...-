import os, re

file_path = r'C:\Users\30568\Desktop\研究生\研1\论文学习\波束资源分配\2026.1~2\论文11复现\main.py'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

new_text, count = re.subn(
    r'selected = np\.where\(F_opt\[s,\s*:\]\s*>\s*0\.5\)\[0\]\.tolist\(\)',
    r'selected = np.argsort(F_opt[s, :])[-4:][::-1].tolist()',
    text
)

if count > 0:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    print(f"Successfully modified main.py (replaced {count} instances)")
else:
    print("Pattern not found!")
