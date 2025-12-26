import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.linalg import eigh
import time

# ==========================================
# 通用函数 (保持不变)
# ==========================================
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_prod(ops):
    return reduce(np.kron, ops)


def get_ising_terms(n, J=1.0, h=1.0):
    terms = []
    for i in range(n - 1):  # ZZ
        s = ['I'] * n
        s[i] = 'Z';
        s[i + 1] = 'Z'
        terms.append((-J, s))
    for i in range(n):  # X
        s = ['I'] * n
        s[i] = 'X'
        terms.append((-h, s))
    return terms


def get_exact_ground_state(n, terms):
    """最耗时的一步: 矩阵对角化"""
    dim = 2 ** n
    # 打印提示，让你知道程序还在活着
    if n > 8: print(f"  ...正在对角化 {dim}x{dim} 矩阵 (只算一次)...")

    H_mat = np.zeros((dim, dim), dtype=complex)
    map_ops = {'I': I, 'X': X, 'Z': Z}
    for coeff, p_list in terms:
        ops = [map_ops[p] for p in p_list]
        H_mat += coeff * kron_prod(ops)
    vals, vecs = eigh(H_mat)
    return vals[0], vecs[:, 0]  # Energy, State


def get_exact_expectation(psi, p_list):
    map_ops = {'I': I, 'X': X, 'Z': Z}
    ops = [map_ops[p] for p in p_list]
    P_mat = kron_prod(ops)
    return np.real(np.vdot(psi, P_mat @ psi))


# ==========================================
# 模拟器 (修改版: 接收预计算的真值)
# ==========================================
def run_simulation_fast(n_qubits, fixed_total_shots, true_E, psi, terms):
    # 这里不再重复计算真值，直接使用传入的 true_E 和 psi

    # 2. 模拟 Standard
    n_terms = len(terms)
    shots_per_term = max(1, fixed_total_shots // n_terms)
    E_std = 0
    for coeff, p_list in terms:
        mu = get_exact_expectation(psi, p_list)
        sigma = np.sqrt(max(0, 1 - mu ** 2))
        noise = np.random.normal(0, sigma / np.sqrt(shots_per_term))
        E_std += coeff * (mu + noise)

    # 3. 模拟 Shadow
    E_shadow = 0
    for coeff, p_list in terms:
        mu = get_exact_expectation(psi, p_list)
        weight = sum(1 for p in p_list if p != 'I')
        shadow_sigma = np.sqrt(3 ** weight - mu ** 2)
        noise = np.random.normal(0, shadow_sigma / np.sqrt(fixed_total_shots))
        E_shadow += coeff * (mu + noise)

    return abs(E_std - true_E), abs(E_shadow - true_E)


# ==========================================
# 主程序: 扫描 N
# ==========================================
if __name__ == '__main__':
    # 稍微调整一下 N 的列表，N=12 可能还是会有点慢，但这次应该能跑完
    n_list = [2, 3, 4, 5, 6, 8, 10, 11]  # 暂时把 12 改成 11，太大的矩阵很吃内存
    FIXED_SHOTS = 20000
    REPEATS = 100

    avg_err_std = []
    avg_err_shadow = []

    print(f"开始运行 N 扫描 (Fixed Shots = {FIXED_SHOTS})...")

    for n in n_list:
        start_t = time.time()

        # 1. 【提速关键】先把这道难题做一次，算出标准答案
        terms = get_ising_terms(n)
        true_E, psi = get_exact_ground_state(n, terms)

        temp_std = []
        temp_shad = []

        # 2. 然后拿着标准答案去跑 100 次模拟 (这一步极快)
        for _ in range(REPEATS):
            # 传入 true_E 和 psi
            e1, e2 = run_simulation_fast(n, FIXED_SHOTS, true_E, psi, terms)
            temp_std.append(e1)
            temp_shad.append(e2)

        mean_std = np.mean(temp_std)
        mean_shad = np.mean(temp_shad)

        avg_err_std.append(mean_std)
        avg_err_shadow.append(mean_shad)

        winner = "Standard" if mean_std < mean_shad else "Shadow"
        elapsed = time.time() - start_t
        print(f"N={n:2d} | Std Err: {mean_std:.4f} | Shad Err: {mean_shad:.4f} | Win: {winner} | Time: {elapsed:.2f}s")

    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, avg_err_std, 'bo-', label='Standard (Error increases with N)', linewidth=2)
    plt.plot(n_list, avg_err_shadow, 'rs-', label='Shadow (Error is stable)', linewidth=2)

    plt.xlabel('Number of Qubits (N)')
    plt.ylabel('Energy RMSE (Lower is Better)')
    plt.title(f'Scalability Test: Ising Model (Total Shots={FIXED_SHOTS})')
    plt.grid(True)
    plt.legend()
    plt.savefig('n_scaling.png', dpi=300)
    print("\n图片已保存为 n_scaling.png")