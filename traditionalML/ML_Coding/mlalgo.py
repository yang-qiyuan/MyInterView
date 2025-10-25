import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Newton's Method
"""
Equation: 
X_new = X_old - f(X_old)/f'(X_old)
"""
def newton_method(tol=1e-8, max_iter=100000):
    x_old =   0

    fprime = 2*x_old
    for _ in range(max_iter):
        x_new = x_old - (x_old**2-2)/(2*x_old+1e-9)
        if abs(x_new - x_old) < tol:
            return x_new
        x_old = x_new
    return x_new

# Gradient Descent
def gradient_descent(lr=0.0001, max_iter=10000000):
    x_old = 10

    for _ in range(max_iter):
        fprime = 2*x_old
        x_new = x_old - fprime*lr
        if abs(x_new - x_old) < 1e-8:
            return x_new
        x_old = x_new
    return x_new

%matplotlib inline
import torch
from d2l import torch as d2l

# Adam
class Adam:
    def __init__(self, lr, t=1, beta1=0.9, beta2=0.999, eps=1e-6):
        self.lr = lr
        self.t = t
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_adam_states(self, feature_dim):
        v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
        s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
        return ((v_w, s_w), (v_b, s_b))

    def adam(self,params, states, hyperparams):
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        for p, (v, s) in zip(params, states):
            with torch.no_grad():
                v[:] = beta1 * v + (1 - beta1) * p.grad
                s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
                v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
                s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
                p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                        + eps)
            p.grad.data.zero_()
        hyperparams['t'] += 1


def tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.
    
    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    # 把二维列表转成字符串列表
    corpus_str = [" ".join(doc) for doc in corpus]

    # 拟合 TF-IDF
    vectorizer = TfidfVectorizer(norm='l2')
    X = vectorizer.fit_transform(corpus_str)

    # 获取词表
    feature_names = vectorizer.get_feature_names_out()
    vocab_index = {word: idx for idx, word in enumerate(feature_names)}

    print(vocab_index)
    print(X.shape)
    # 构建结果
    scores = []
    for doc_idx in range(X.shape[0]):
        doc_scores = []
        for word in query:
            if word in vocab_index:
                score = X[doc_idx, vocab_index[word]]
            else:
                score = 0.0
            doc_scores.append(round(float(score), 5))  # 保留5位小数，方便对齐示例
        scores.append(doc_scores)

    return scores


def decision_tree(data):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    """
    data input
    [ [50000, 1, 'Yes', 'Good'], [50000, 2, 'No', 'Bad'], [70000, 3, 'Yes', 'Good'], [40000, 4, 'No', 'Bad'] ]
    """

    data = eval(data)
    l1 = LabelEncoder()
    l2 = LabelEncoder()
    labels = [d[-1] for d in data]
    f3 = [d[-2] for d in data]
    lb1 = l1.fit_transform(labels)
    feature3 = l2.fit_transform(f3)

    for idx, d in enumerate(data):
        data[idx][-1] = lb1[idx]
        data[idx][-2] = feature3[idx]

    features = [d[:-1] for d in data]
    labels = [d[-1] for d in data]

    clf = DecisionTreeClassifier(random_state=42)
    predicted = clf.fit(features, labels)
    feature_importances = predicted.feature_importances_
    return np.argmax(feature_importances)


def svd(train, test):
    """
    PCA（仅保留第一主成分）压缩-重建并计算测试集样本的 MSE。

    输入：
    - train: 二维列表/数组，形状 (n_samples, m_features)
    - test : 二维列表/数组，形状 (k_samples, m_features)

    步骤：
    1) 去均值（仅用训练集求均值 μ），所有样本都减去 μ 得到 Xc
    2) 协方差矩阵（总体方差，ddof=0）：Sigma = (1/n) * Xc.T @ Xc
    3) 特征分解（np.linalg.eigh），按特征值从大到小取第一特征向量 vmax
    - 方向标准化：若 vmax 的首个非零分量为负，则乘以 -1 以固定方向
    4) 投影-重建：
    - z = (x - μ)^T vmax
    - x_hat = μ + z * vmax
    5) 对每个测试样本计算 MSE(x) = (1/m) * sum((x - x_hat)^2)
    6) 输出：将所有测试样本的 MSE 保留两位小数（字符串），组成 JSON 数组（一行）
    """
    # train and test a 2-D array
    # test is also a 2-D array
    train = np.array(train, dtype=float)
    test = np.array(test, dtype=float)
    mu = np.mean(train, axis=0)
    x_c = train - mu
    covar = (x_c.T@x_c)/len(train)
    eigenvalue, eigenvector = np.linalg.eigh(covar)
    v_max = eigenvector[:,np.argmax(eigenvalue)]

    # 方向准则化
    first_non_zero_vect = np.nonzero(v_max)[0]
    if len(first_non_zero_vect) > 0:
        target = first_non_zero_vect[0]
        if v_max[target] < 0:
            v_max = - v_max
    
    mse = []
    for x in test:
        x_center = x - mu
        z = x_center.T@v_max
        x_hat = mu + z*v_max
        res = np.mean((x - x_hat)**2,axis=0)
        mse.append(f"{res:.2f}")
    import json
    print(json.dumps(mse))

def k_nearest_neighbors(X, y, test_sample, k):
    """
    X: 二维列表/数组，形状 (n_samples, m_features)
    y: 一维列表/数组，形状 (n_samples,) -- 每个样本的标签
    test_sample: 一维列表/数组，形状 (m_features,)
    k: 整数
    
    步骤：
    1) 计算测试样本与训练样本的距离
    2) 找到距离最近的k个样本
    3) 返回距离最近的k个样本的标签
    4) 返回距离最近的k个样本的标签中出现次数最多的标签
    5) 返回距离最近的k个样本的标签中出现次数最多的标签
    """
    # get the l2 norm (euclidean distance)
    dist = np.linalg.norm(X-test_sample.T, axis=1)
    first_k = np.argsort(dist)[:k]
    nei = y[first_k]
    vals, cnts = np.unique(nei, return_counts=True)
    return vals[np.argmax(cnts)]


if __name__ == "__main__":
    # corpus = [["hello", "world"], ["hello", "python"]]
    # query = ["hello", "python"]
    # print(tf_idf(corpus, query))
    # data = """[[50000, 1, 'Yes', 'Good'], [50000, 2, 'No', 'Bad'], [70000, 3, 'Yes', 'Good'], [40000, 4, 'No', 'Bad']]"""
    # print(decision_tree(data))
    matrix = np.array([
    [2, -1, -1, 0, 0, 0],
    [-1, 2, -1, 0, 0, 0],
    [-1, -1, 3, -1, 0, 0],
    [0, 0, -1, 3, -1, -1],
    [0, 0, 0, -1, 2, -1],
    [0, 0, 0, -1, -1, 2]
])
    eigenvalue, eigenvector = np.linalg.eigh(matrix)
    # eigenvector = eigenvector[:,np.argmax(eigenvalue)]
    print(eigenvalue)
    print(eigenvector)
  