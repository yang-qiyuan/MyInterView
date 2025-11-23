# Linear Discriminant Analysis (LDA)
1. Steps to perform LDA
```Text
a. Maximize the distance between means of each category.
b. Minimize the variation within each category.

```

## 线性判别分析 (LDA) 公式

### 1. 类别均值
对于类别 $i$，有 $N_i$ 个样本：

$$
\mu_i = \frac{1}{N_i} \sum_{x \in C_i} x
$$

总体均值：

$$
\mu = \frac{1}{N} \sum_{i=1}^{k} \sum_{x \in C_i} x
$$

### 2. 类内散布矩阵
$$
S_W = \sum_{i=1}^{k} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T
$$

### 3. 类间散布矩阵
$$
S_B = \sum_{i=1}^{k} N_i (\mu_i - \mu)(\mu_i - \mu)^T
$$

### 4. 广义特征值问题
通过解以下问题获得判别向量：
$$
S_W^{-1} S_B w = \lambda w
$$

### 5. 投影
选择最大特征值对应的特征向量 $w_1, w_2, \dots, w_m$，组成投影矩阵：

$$
W = [w_1, w_2, \dots, w_m]
$$

对样本 $x$ 的投影为：

$$
y = W^T x
$$
