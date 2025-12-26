import numpy as np
import time
from pyquil import Program, get_qc
from pyquil.gates import *
from helper import propagate, ansatz


# ==========================================
# 1. 计数器类
# ==========================================
class MetricsTracker:
    def __init__(self):
        self.total_shots = 0
        self.total_gates = 0
        self.circuit_depth_history = []
        self.step_times = []


tracker = MetricsTracker()


# ==========================================
# 2. 极速通用版: 采集经典阴影 (Smart Batching)
# ==========================================
def collect_shadows(alist, shots, qc_name, qbits):
    """
    采集经典阴影 (针对单比特和多比特都进行了极致优化)
    原理: 将相同的测量基底组合 (Basis Configuration) 打包在一起运行
    """
    start_time = time.time()
    qc = get_qc(qc_name) if isinstance(qc_name, str) else qc_name
    n_qubits = len(qbits)
    snapshots = []

    # 统计电路深度 (仅做一次)
    temp_p = Program()
    ansatz(temp_p, qbits)
    propagate(temp_p, alist, qbits)
    tracker.total_gates += len(temp_p.instructions) * shots
    tracker.circuit_depth_history.append(len(temp_p.instructions))

    # 1. 生成所有需要的随机基底
    # shape: (shots, n_qubits), 元素为 0, 1, 2
    all_bases = np.random.randint(0, 3, size=(shots, n_qubits))

    # 2. 【核心优化】对基底进行“分组”
    # 我们用字典统计每种基底配置出现了多少次
    # 例如: (0, 2, 1, 0) 出现了 25 次
    basis_groups = {}
    for row in all_bases:
        key = tuple(row)  # 转成 tuple 才能做字典的 key
        basis_groups[key] = basis_groups.get(key, 0) + 1

    # 3. 遍历每一种基底配置 (对于 4 比特，最多只有 81 组，远小于 2000)
    for basis_tuple, count in basis_groups.items():
        if count == 0: continue

        # 构建电路
        p = Program()
        p.wrap_in_numshots_loop(count)  # <--- 一次跑 count 次，极大减少网络通讯
        ro = p.declare('ro', 'BIT', n_qubits)

        ansatz(p, qbits)
        propagate(p, alist, qbits)

        # 根据当前的基底配置添加旋转门
        for i, basis_val in enumerate(basis_tuple):
            target_q = qbits[i]
            if basis_val == 0:  # Measure X
                p += RY(-np.pi / 2, target_q)
            elif basis_val == 1:  # Measure Y
                p += RX(np.pi / 2, target_q)
            # basis_val 2 is Z (Identity)

        # 添加测量
        for k, q in enumerate(qbits):
            p += MEASURE(q, ro[k])

        # 编译并运行 (最耗时的步骤，现在次数大大减少)
        exe = qc.compile(p)
        res = qc.run(exe)

        # 提取结果
        if hasattr(res, 'readout_data'):
            raw_data = res.readout_data['ro']
        else:
            raw_data = res

        # raw_data shape: (count, n_qubits)
        # 转换为 +1/-1
        vals_batch = np.where(np.array(raw_data) == 0, 1, -1)

        # 将这一批结果存入总列表
        # snapshot 格式: (basis_list, outcome_list)
        basis_list = list(basis_tuple)
        for i in range(count):
            snapshots.append((basis_list, vals_batch[i]))

    tracker.total_shots += shots
    tracker.step_times.append(time.time() - start_time)

    return snapshots


# ==========================================
# 3. 估算函数 (保持不变)
# ==========================================
def estimate_shadow_observable(snapshots, pauli_info):
    # 兼容单比特整数索引
    if isinstance(pauli_info, int):
        if pauli_info == 0: return 1.0
        target_basis = pauli_info - 1
        match_sum = 0
        for bases, outcomes in snapshots:
            if bases[0] == target_basis:
                match_sum += 3 * outcomes[0]
        return match_sum / len(snapshots)

    # 多比特 Pauli String 列表 [(q, type), ...]
    pauli_list = pauli_info
    if not pauli_list: return 1.0

    k = len(pauli_list)
    match_sum = 0

    for bases, outcomes in snapshots:
        hit = True
        product_val = 1
        for (q_idx, p_type) in pauli_list:
            needed_basis = p_type - 1
            if bases[q_idx] != needed_basis:
                hit = False
                break
            product_val *= outcomes[q_idx]

        if hit:
            match_sum += product_val * (3 ** k)

    return match_sum / len(snapshots)


# ==========================================
# 4. 能量计算 (保持不变)
# ==========================================
def calc_shadow_energy(snapshots, hm_list):
    E = 0
    for term in hm_list:
        if isinstance(term, tuple):
            coeff, pauli_list = term
            exp_val = estimate_shadow_observable(snapshots, pauli_list)
            E += coeff * exp_val
    return E