import os

file_path = 'algorithms/solvers.py'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

old_str = '''            try:
                prob.solve(solver=cp.SCS, warm_start=True)
                if F_var.value is not None:
                    F_best = F_var.value
                    alpha += 2 * beta * F_best * (1 - F_best)
                    beta *= rho
                else:'''

new_str = '''            try:
                prob.solve(solver=cp.SCS, warm_start=True)
                if F_var.value is not None:
                    F_best = F_var.value
                    print(f"        [Theta {_}] Current F_best sum: {np.sum(F_best):.2f}, Max Val: {np.max(F_best):.2f}")
                    alpha += 2 * beta * F_best * (1 - F_best)
                    beta *= rho
                else:'''

if old_str in text:
    text = text.replace(old_str, new_str)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully added F_best print to solvers.py")
else:
    print("Could not find Target String.")
