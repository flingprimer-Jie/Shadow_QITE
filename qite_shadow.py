import numpy as np
import matplotlib.pyplot as plt
import time
from helper_shadow import collect_shadows, calc_shadow_energy, estimate_shadow_observable, tracker
from helper import update_alist
from models import HamiltonianFactory  # 导入之前写好的模型工厂


# =========================================================
# QITE 单步演化 (仅适用于单量子比特)
# =========================================================
def qite_step_shadow_single_qubit(alist, snapshots, db, delta, hm_list):
    """
    专门用于单比特的更新逻辑 (依赖 4x4 矩阵求解)
    """
    # 1. 计算 I, X, Y, Z 的期望值
    sigma_expectation = np.zeros([4], dtype=complex)
    for j in range(4):
        # j=0->I, j=1->X... 传入整数索引
        sigma_expectation[j] = estimate_shadow_observable(snapshots, j)

    # 2. 更新参数
    # 注意：这里需要把 models.py 的通用格式转回 update_alist 能懂的旧格式
    # models.py: (coeff, [(0, 1)]) -> X
    # helper.py 需要: hm_list[j] 这种复杂结构
    # 为了简化，我们在这里手动为 update_alist 构造它需要的 hm 输入

    # 这是一个临时的桥接，因为 update_alist 太旧了
    # 我们遍历 hm_list，找到对应的项来更新
    for term in hm_list:
        coeff, pauli_list = term

        # 构造一个 update_alist 能懂的虚拟 hm 对象
        # 格式: [[indices], [coeffs]]
        # 这里只处理简单的 X, Y, Z 项
        if len(pauli_list) == 1:
            p_type = pauli_list[0][1]  # 1, 2, 3
            # 构造 helper.py 期望的格式: [ [[p_type], [coeff]] ]
            legacy_hm = [[[p_type], [coeff]]]
            update_alist(sigma_expectation, alist, db, delta, legacy_hm)

    return alist


# =========================================================
# 主程序
# =========================================================
def qite_shadow_main(model_name, qc_name, shots, db, delta, N):
    # 1. 加载模型
    print(f"\nLoading Model: {model_name}...")
    hm_list, n_qubits = HamiltonianFactory.get_model(model_name)
    qbits = list(range(n_qubits))
    print(f"Model loaded. Qubits: {n_qubits}, Terms: {len(hm_list)}")

    E = np.zeros([N + 1], dtype=complex)
    alist = []

    # 2. 初始化
    print(f"Step 0/{N}: Measuring Initial Energy...")
    current_snapshots = collect_shadows(alist, shots, qc_name, qbits)
    E[0] = calc_shadow_energy(current_snapshots, hm_list)
    print(f"  Energy: {np.real(E[0]):.4f}")

    # 3. 演化循环
    for i in range(1, N + 1):
        print(f"Step {i}/{N}...")

        # A. 采集阴影 (所有模型通用)
        current_snapshots = collect_shadows(alist, shots, qc_name, qbits)

        # B. 参数更新 (Update)
        if n_qubits == 1:
            # 单比特模型：可以跑完整的 QITE 更新
            alist = qite_step_shadow_single_qubit(alist, current_snapshots, db, delta, hm_list)
        else:
            # 多比特模型：跳过更新 (因为 update_alist 不支持)
            if i == 1: print(
                "  [Warning] Multi-qubit QITE update is skipped (requires generic solver). Measuring energy only.")
            # alist 保持不变，相当于在一个固定的 Ansatz 上反复测量
            pass

        # C. 计算能量 (Energy)
        # 为了画图准确，重新采集一次 (也可以复用 current_snapshots 只要 alist 没变)
        if n_qubits == 1:
            new_snapshots = collect_shadows(alist, shots, qc_name, qbits)
            E[i] = calc_shadow_energy(new_snapshots, hm_list)
        else:
            # 多比特未更新，直接复用上一步的能量，或者是测量的波动
            # 为了展示阴影的测量效果，我们还是重测一次
            new_snapshots = collect_shadows(alist, shots, qc_name, qbits)
            E[i] = calc_shadow_energy(new_snapshots, hm_list)

        print(f"  Energy: {np.real(E[i]):.4f}")

    return E


# =========================================================
# 启动入口
# =========================================================
if __name__ == '__main__':
    # --- 设置参数 ---
    # 可选模型: 'single_qubit_test', 'heisenberg', 'lih_molecule', 'ising'
    MODEL_NAME = 'ising'  # <-- 在这里改名字即可！

    # qc = '1q-qvm'  # 如果跑多比特，记得确保 Docker 里开了足够大的 QVM
    qc = '4q-qvm'  # <--- 改成 4q (或者 9q-qvm 也可以)，以支持多比特模型
    shots = 2000
    N = 25
    db = 0.1
    delta = 0.1

    global_start = time.time()

    # 运行
    E = qite_shadow_main(MODEL_NAME, qc, shots, db, delta, N)

    global_end = time.time()

    # --- 报告 ---
    print("\n" + "=" * 30)
    print("      Analysis Report      ")
    print("=" * 30)
    print(f"Total Time: {global_end - global_start:.2f} s")
    print(f"Total Shots: {tracker.total_shots}")
    print(f"Total Gates: {tracker.total_gates}")

    # --- 画图 ---
    plt.plot(np.arange(0, N + 1) * db, np.real(E), 'bo-', label=f'{MODEL_NAME}')
    plt.grid()
    plt.legend()
    plt.title(f"QITE Shadow: {MODEL_NAME}")
    plt.ylabel("Energy")
    plt.xlabel("Imaginary Time")
    plt.show()