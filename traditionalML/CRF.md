# Conditional Random Field (CRF)

1. Steps to perform CRF

```Text
a. Define features over input–output pairs.
b. Compute potential functions for each node and edge.
c. Train parameters by maximizing conditional log-likelihood.
d. Use dynamic programming (e.g., Viterbi) for inference.
```

## 条件随机场 (CRF) 公式

### 1. 条件概率定义

给定输入序列 $X = (x_1, x_2, \dots, x_n)$，输出标签序列 $Y = (y_1, y_2, \dots, y_n)$ 的条件概率为：

$$
P(Y|X) = \frac{1}{Z(X)} \exp\left( \sum_{i} \sum_{k} \lambda_k f_k(y_{i-1}, y_i, X, i) \right)
$$

其中：

* $f_k(y_{i-1}, y_i, X, i)$ 是第 $k$ 个特征函数
* $\lambda_k$ 是其对应的权重
* $Z(X)$ 是归一化因子（配分函数）：

$$
Z(X) = \sum_{Y} \exp\left( \sum_{i} \sum_{k} \lambda_k f_k(y_{i-1}, y_i, X, i) \right)
$$

---

### 2. 目标函数（对数似然）

训练时最大化条件对数似然：

$$
\mathcal{L}(\lambda) = \sum_{j=1}^{N} \log P(Y^{(j)} | X^{(j)}) - \frac{1}{2\sigma^2} |\lambda|^2
$$

第一项为观测数据的对数似然，第二项为L2正则化。

---

### 3. 梯度

每个参数 $\lambda_k$ 的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial \lambda_k}
= \sum_{j=1}^{N} \left( \sum_{i} f_k(y_{i-1}^{(j)}, y_i^{(j)}, X^{(j)}, i)

* \mathbb{E}*{P(Y|X^{(j)})}[f_k(y*{i-1}, y_i, X^{(j)}, i)] \right)
* \frac{\lambda_k}{\sigma^2}
  $$

---

### 4. 推断 (Inference)

最优标签序列 $\hat{Y}$：

$$
\hat{Y} = \arg\max_{Y} P(Y|X)
$$

通常使用 **Viterbi 算法** 动态规划求解。

---

### 5. 特点总结

* 与HMM相比，CRF直接建模 $P(Y|X)$，避免了独立性假设。
* 能同时利用上下文特征和全局约束。
* 训练较慢但泛化性能强。

