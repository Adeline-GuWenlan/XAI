可以的，只要注意两点：**维度增长很快**、**阈值会跟着改变**。下面给出适用于任意 $n$ 比特的通用实现，并演示如何分别生成 3、4、5 个比特的数据集。

---

## 为什么阈值会变化？

稳定子范数定义为：

$$
\|A\|_{\mathrm{st}}=\frac{1}{2^n}\sum_{P\in P_n}\bigl|\mathrm{Tr}(A P)\bigr|,\quad P_n\text{为所有$n$比特Pauli张量积。}
$$

对于密度矩阵 $\rho$，$\mathrm{Tr}(\rho)=1$，因此

$$
\|\,\rho\|_{\mathrm{st}}
=\frac{1}{2^n}\Bigl(1+\underbrace{\sum_{P\neq I}|\mathrm{Tr}(\rho\,P)|}_{\mathrm{SN}_n}\Bigr)
\le 1
\quad\Longleftrightarrow\quad
\mathrm{SN}_n \le 2^n-1.
$$

上式说明：如果用 **未归一化** 的 $\mathrm{SN}_n$（所有非恒等 Pauli 张量积的迹绝对值之和）作为标签，则无魔性的阈值是 $2^n-1$；如果按照定义返回 $\|\,\rho\|_{\mathrm{st}}$，则阈值恒为 1。

例如：

| qubits | 非归一化SN阈值 | 归一化$D(\rho)$阈值 | 代码中返回值（示例）      |
| ------ | -------- | -------------- | --------------- |
| 2      | 3        | 1              | `sn/4`≤0.75     |
| 3      | 7        | 1              | `sn/8`≤0.875    |
| 4      | 15       | 1              | `sn/16`≤0.9375  |
| 5      | 31       | 1              | `sn/32`≤0.96875 |

---

## 通用的 `get_sn` 函数

为了避免手工列举所有组合，可以利用进制转换遍历所有长度为 `qubits` 的四进制串来索引 $\{I,X,Y,Z\}$ 的张量积。下面是一个更通用的实现，在 `qubits>=1` 时都能正确计算：

```python
import numpy as np

# 单比特Pauli矩阵
I = np.array([[1,0],[0,1]], dtype=np.complex128)
X = np.array([[0,1],[1,0]], dtype=np.complex128)
Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
Z = np.array([[1,0],[0,-1]], dtype=np.complex128)
paulis = [I, X, Y, Z]

def number_to_base(n, b, length):
    """把整数n转换为length位的b进制列表，不足位补0。"""
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    while len(digits) < length:
        digits.append(0)
    return digits[::-1]

def get_sn_general(rho, qubits):
    """
    返回 (1/2^qubits) * sum_{非恒等P} |Tr(rho P)|
    也就是归一化的稳定子范数 D(rho) 减去 1/2^n 常数。
    如果想得到未归一化的SN_n，只需删除最后的除以 2**qubits。
    """
    if qubits == 1:
        # 与之前相同
        rx = np.trace(rho @ X).real
        ry = np.trace(rho @ Y).real
        rz = np.trace(rho @ Z).real
        return (abs(rx) + abs(ry) + abs(rz)) / 2**1
    else:
        sn = 0.0
        dim_pauli = 4 ** qubits
        for idx in range(dim_pauli):
            # 将 idx 展开成 qubits 位四进制，表示第一个比特用paulis[d0]，第二个用paulis[d1]...
            digits = number_to_base(idx, 4, qubits)
            if all(d == 0 for d in digits):
                continue  # 跳过 I⊗I⊗...⊗I
            # 构造张量积 Pauli 矩阵
            sigma = paulis[digits[0]]
            for d in digits[1:]:
                sigma = np.kron(sigma, paulis[d])
            sn += abs(np.trace(rho @ sigma))
        # 返回 D(rho) - 1/2^n = sn/2**n （方便直接与阈值比较）
        return sn / (2 ** qubits)
```

* 如果你希望返回未归一化的 $\mathrm{SN}_n$（即所有非恒等项的绝对值之和），只需去掉最后的除以 `2 ** qubits`。
* 对于单比特，上述函数返回的是 $\frac{|r_x|+|r_y|+|r_z|}{2}$，而 $\|\,\rho\|_{\mathrm{st}} = 0.5(1 + \mathrm{SN}_1)$；这与文献定义相符。

---

## 修改 `generate_dataset` 支持 3、4、5 比特

```python
def generate_dataset(num_samples, qubits, lam, output_path=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    D = 2 ** qubits
    states = []
    labels = []
    for i in range(num_samples):
        rho = random_mixed_state_poisson(D, lam)
        rho /= np.trace(rho)
        # 使用通用 SN 计算
        sn_label = get_sn_general(rho, qubits)
        states.append(rho)
        labels.append(sn_label)
    states = np.array(states)
    labels = np.array(labels)
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        np.save(os.path.join(output_path, f"{qubits}q_{num_samples}_states.npy"), states)
        np.save(os.path.join(output_path, f"{qubits}q_{num_samples}_labels.npy"), labels)
    return states, labels
```

---

## 运行示例：分别生成 3、4、5 比特数据集

```python
if __name__ == "__main__":
    for n in [3, 4, 5]:
        states, labels = generate_dataset(num_samples=200, qubits=n, lam=10.0,
                                          output_path=f"data_{n}q")
        # 无魔性阈值为 (2**n - 1)/(2**n)，例如 n=3 时为 7/8≈0.875
        threshold = (2**n - 1) / (2**n)
        below = np.sum(labels <= threshold)
        above = len(labels) - below
        print(f"== {n} qubits ==")
        print(f"threshold D(rho) = {threshold:.3f}")
        print(f"labels <= threshold: {below} ({below/len(labels)*100:.1f}%)")
        print(f"labels > threshold: {above} ({above/len(labels)*100:.1f}%)")
        print()
```

这段代码会分别生成 3、4、5 比特的样本，并按照文献中的阈值判断有无魔性。请注意：

* **计算量迅速增大**：3 比特需要计算 63 个 Pauli 项，4 比特需要 255 个，5 比特则 1023 个。样本数过大时运行会比较慢。
* **阈值随比特数升高而增大**：上面计算的 `threshold` 就是归一化稳定子范数应小于等于 1 时对应的界限；如果你使用未归一化和不除以 `2**n` 的结果，则阈值是 $2^n - 1$。请根据自己代码返回值的定义来选择比较标准。

这样修改后，代码即可独立运行 3、4、5 比特情况，并能生成对应的数据集。
