import numpy as np
import qutip
import pandas as pd
from tqdm import tqdm

# --- 您提供的函数 ---
def vec_to_rho(vec: np.ndarray) -> np.ndarray:
    """
    反展开：把向量还原成 Hermitian 矩阵。
    """
    # 对于两比特系统，D=4，向量长度应为16。
    D = 4 
    if len(vec) != D * D:
        # 您可以根据需要调整这里的错误处理
        print(f"警告: 输入向量长度为 {len(vec)}，但期望为 {D*D}。跳过此行。")
        return None
        
    rho = np.zeros((D, D), dtype=np.complex128)
    k = 0
    for i in range(D):
        # 对角元
        rho[i, i] = vec[k]
        k += 1
        # 非对角元
        for j in range(i + 1, D):
            rho[i, j] = vec[k] + 1j * vec[k + 1]
            rho[j, i] = rho[i, j].conj()
            k += 2
    return rho
def compute_core_features(rho_matrix: np.ndarray) -> dict:
    """
    精简指标集（两比特）：
    - S_rho_bits：冯诺依曼熵 S(ρ) [bit]
    - purity_scalar：纯度 Tr(ρ^2)
    - Hinf_bits：极小熵 H_∞ = -log2(λ_max)
    - r_bloch_A, s_bloch_B：局域Bloch向量（各3维）
    - T_corr：3x3 相关矩阵 T_ij = Tr[ρ (σ_i ⊗ σ_j)]
    - CHSH_bound_max：Horodecki 上界 2*sqrt(t1^2 + t2^2)
    - T_fro_norm：||T||_F
    - C_rel_bits：相干的相对熵 [bit]
    - C_conc：Concurrence（两比特纠缠度）
    - N_neg：Negativity
    """
    # ——— 建立 Qobj（声明两比特维度） ———
    rho_q = qutip.Qobj(np.asarray(rho_matrix, dtype=np.complex128),
                       dims=[[2, 2], [2, 2]])

    # ========== A. 熵/混合度 ==========
    # 冯诺依曼熵 S(ρ) in bits
    S_rho_bits = float(qutip.entropy_vn(rho_q, base=2))

    # 纯度 Tr(ρ^2)
    purity_scalar = float((rho_q ** 2).tr().real)

    # 极小熵 H_inf = -log2(λ_max)
    evals = np.linalg.eigvalsh(rho_q.full())
    # 数值修正到[0,1]
    evals = np.clip(np.real_if_close(evals), 0.0, 1.0)
    lam_max = float(np.max(evals))
    # 防止 log(0)
    eps = 1e-15
    Hinf_bits = float(-np.log2(max(lam_max, eps)))

    # ========== C. 保利展开：r, s, T ==========
    sig = [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
    I2 = qutip.qeye(2)

    # Bloch 向量（A、B）
    r_bloch_A = np.array([qutip.expect(qutip.tensor(s, I2), rho_q) for s in sig], dtype=float)
    s_bloch_B = np.array([qutip.expect(qutip.tensor(I2, s), rho_q) for s in sig], dtype=float)

    # 3x3 相关矩阵 T
    T_corr = np.array([[qutip.expect(qutip.tensor(si, sj), rho_q) for sj in sig] for si in sig], dtype=float)

    # T 的派生量：CHSH 上界与 Fro 范数
    t_svals = np.linalg.svd(T_corr, compute_uv=False)
    # 奇异值降序
    t_svals = np.sort(t_svals)[::-1]
    t1, t2 = float(t_svals[0]), float(t_svals[1])
    CHSH_bound_max = float(2.0 * np.sqrt(t1 ** 2 + t2 ** 2))
    T_fro_norm = float(np.linalg.norm(T_corr, ord='fro'))

    # ========== D. 相干（相对熵） ==========
    rho_full = rho_q.full()
    rho_diag = np.diag(np.diag(rho_full))
    rho_diag_q = qutip.Qobj(rho_diag, dims=[[2, 2], [2, 2]])
    C_rel_bits = float(qutip.entropy_vn(rho_diag_q, base=2) - S_rho_bits)

    # ========== E. 纠缠 ==========
    C_conc = float(qutip.concurrence(rho_q))
    # Negativity：N = (||ρ^{T_B}||_1 - 1)/2
    rho_PT = qutip.partial_transpose(rho_q, [0, 1], [False, True])  # 对 B 取部分转置
    evals_PT = np.linalg.eigvalsh(rho_PT.full())
    N_neg = float((np.sum(np.abs(evals_PT)) - 1.0) / 2.0)

    return {
        # 熵/混合度
        "S_rho_bits": S_rho_bits,          # 冯诺依曼熵 S(ρ) [bit]
        "purity_scalar": purity_scalar,    # 纯度 Tr(ρ^2)
        "Hinf_bits": Hinf_bits,            # 极小熵 H_∞ [bit]

        # 保利展开结构
        "r_bloch_A": r_bloch_A,            # A 边 Bloch 向量 (3,)
        "s_bloch_B": s_bloch_B,            # B 边 Bloch 向量 (3,)
        "T_corr": T_corr,                  # 相关矩阵 T (3x3)

        # 由 T 派生
        "CHSH_bound_max": CHSH_bound_max,  # Horodecki CHSH 上界
        "T_fro_norm": T_fro_norm,          # ||T||_F

        # 相干
        "C_rel_bits": C_rel_bits,          # 相干相对熵 [bit]

        # 纠缠
        "C_conc": C_conc,                  # concurrence
        "N_neg": N_neg,                    # negativity
    }

# ==============================================================================
# --- 主程序：加载数据并进行批量处理 ---
# ==============================================================================

# V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V
# --- 请在这里修改您的输入文件名 ---
INPUT_NPY_FILE = 'Rawdata/Input/input_for_2_qubits_mixed_2_50000_datapoints.npy' 
OUTPUT_CSV_FILE = 'physical_quantities_output_50000.csv'
# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

try:
    # 1. 加载包含所有向量的大型 .npy 文件
    print(f"正在加载数据从: {INPUT_NPY_FILE}...")
    all_vectors = np.load(INPUT_NPY_FILE)
    print(f"加载成功！数据形状: {all_vectors.shape}")
    num_samples = all_vectors.shape[0]
except FileNotFoundError:
    print(f"错误: 文件 '{INPUT_NPY_FILE}' 未找到。请检查文件名和路径。")
    exit()

# 2. 准备一个列表来存储每一行的计算结果
all_results = []

# 3. 遍历 .npy 文件中的每一行
print(f"开始处理 {num_samples} 个样本...")
for i in tqdm(range(num_samples), desc="Processing density matrices"):
    # 获取当前行的向量
    vec = all_vectors[i, :]
    
    # 将向量转换为密度矩阵
    rho_matrix = vec_to_rho(vec)
    
    # 如果转换成功，则计算物理量
    if rho_matrix is not None:
        physical_data = compute_core_features(rho_matrix)
        
        if physical_data is not None:
            # 添加一个样本ID，方便追踪
            physical_data['sample_id'] = i
            all_results.append(physical_data)

# 4. 将所有结果转换为一个 Pandas DataFrame
print("处理完成，正在生成DataFrame...")
features_df = pd.DataFrame(all_results)
if not features_df.empty:
    features_df.set_index('sample_id', inplace=True)

    # 5. 将 DataFrame 保存到 CSV 文件
    features_df.to_csv(OUTPUT_CSV_FILE)

    print(f"\n成功！所有物理量已保存到 '{OUTPUT_CSV_FILE}'")
    print("DataFrame 的前5行内容:")
    print(features_df.head())
else:
    print("处理完成，但没有生成任何有效数据。请检查输入文件或vec_to_rho函数。")