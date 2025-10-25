20250528面试记录

一个挺好的男领导面试的,妈的,终于是正常时间面试了,所以还是请了半天假

二面也只有一个人,我服了..

本喵被狠狠的拷打

1. 讲一下模型微调项目细节,是解决什么场景下这个问题?
2. 和rag有什么区别?为什么不用rag?
3. 微调结果怎么去评判他回答的好或者不好?
4. 你在最后这家公司你觉得是你理想的工作状态吗?
5. 讲一下我的另外一个项目...省略
6. 怎么优化这个视觉检测项目?
7. 有没有一个某一个东西是你完全从嗯嗯理念想法或者概念是你主动特别想做的然后你去推动主动去推动去做的
8. 说一说你反洗钱模型怎么做的?就是业务人员有什么经验什么是违规的交易/什么是洗钱交易?怎么把历史的业务经验转化为模型训练代码呢?
   ```
   https://grok.com/chat/96d65bde-9616-4c4d-981a-e90d6d11c9f1  [重要]

    把这种经验用sql查出来作为参数送进去

    常见的违规/洗钱交易特征
   - **高频小额交易（分拆交易，Smurfing）**：
     - 客户在短时间内进行多次小额交易，试图规避监管阈值（例如，低于报告要求的金额）。
     - 例：一天内多次转账，每次金额略低于1万元人民币（中国监管要求）。

   - **异常资金流向**：
     - 资金流向与客户身份或业务背景不符。例如，低收入客户突然有大额资金流入，或资金流向高风险国家/地区。
     - 例：个人账户频繁与空壳公司或高风险行业（如博彩、虚拟货币）相关账户交易。

   - **快速资金流转**：
     - 资金在账户间快速转入转出，停留时间短，缺乏合理的商业目的。
     - 例：资金在多个账户间快速循环，最终流向未知账户。

   - **异常交易模式**：
     - 与客户历史交易模式不符的交易。例如，长期低活跃账户突然出现大额交易。
     - 使用多个账户进行复杂、迂回的资金转移，掩盖资金来源。

   - **可疑身份信息**：
     - 客户身份信息不完整、虚假，或账户实际使用人与登记信息不一致。
     - 例：账户注册为个人，但交易行为显示为企业用途。

   - **高风险行业或产品**：
     - 涉及虚拟货币、贵金属、跨境支付、现金密集型行业（如赌场、典当行）等的交易。
     - 例：频繁使用虚拟货币兑换法定货币。

   - **地理风险**：
     - 资金流向或来源于被列入反洗钱高风险名单的国家/地区。
     - 例：FATF（金融行动特别工作组）灰名单国家的交易。


      将业务经验量化为特征的常见方法

     - **交易特征**：
     - 交易金额、频率、时间间隔。
     - 交易对手数量、账户类型（个人/企业）。
     - 资金流转速度：资金在账户间的停留时间。
     - 交易模式：是否为分拆交易（例如，多次低于阈值的交易）。
   - **客户行为特征**：
     - 与历史交易模式的偏差：使用统计指标（如均值、标准差）计算交易金额/频率的变化。
     - 客户风险评分：基于KYC信息，结合职业、收入、注册地等计算风险分数。
   - **网络特征**（适用于GNN）：
     - 账户间的交易网络：构建图结构，节点为账户，边为交易，边权重为交易金额。
     - 社区检测：识别交易网络中的异常子图（如洗钱团伙）。
   - **时间序列特征**（适用于LSTM）：
     - 交易时间序列：将交易金额、频率按时间窗口（小时、天、周）聚合。
     - 异常波动：检测交易金额或频率的突变。
   - **外部风险特征**：
     - 地理风险：交易对手所在国家/地区的风险评分（参考FATF名单）。
     - 行业风险：交易涉及行业的风险等级。


   随机森林/梯度提升（GBDT，如XGBoost、LightGBM）
   适用场景：基于结构化数据（如交易金额、频率）检测可疑交易。
   优点：可解释性强，适合规则驱动的场景。

   LSTM：
   适用场景：检测时间序列中的异常模式（如资金快速流转）

   GNN（图神经网络）：
   适用场景：检测交易网络中的洗钱团伙或复杂资金流。
   训练流程：
   构建交易图：节点为账户，边为交易。
   提取节点特征（账户属性）和边特征（交易金额、时间）。
   使用GNN（如GraphSAGE、GAT）学习节点嵌入，检测异常节点或子图。
   评估：结合社区检测算法（如Louvain）验证团伙交易。
     

   用随机森林初步筛选可疑交易。
   用LSTM分析时间序列中的异常模式。
   用GNN检测交易网络中的团伙行为。
   ```
   | 特征名称               | 描述                                   | 业务依据                     |
   |------------------------|----------------------------------------|------------------------------|
   | avg_transaction_amount | 账户过去30天的平均交易金额             | 检测异常大额交易             |
   | transaction_frequency  | 每日交易次数                           | 识别分拆交易                 |
   | is_high_risk_country   | 交易对手是否在高风险国家               | 符合FATF监管要求             |
   | account_age            | 账户开立时间                           | 新账户可能风险更高           |
   | network_centrality     | 账户在交易网络中的中心性（GNN计算）    | 识别洗钱团伙核心账户         |
   
   | 模型类型       | 解释性方法                     | 优点                              | 局限性                           |
   |----------------|-------------------------------|-----------------------------------|-----------------------------------|
   | **随机森林/XGBoost** | 特征重要性、SHAP、LIME、规则提取 | 解释性强，易于生成规则             | 限于单条交易，难捕捉序列/网络模式 |
   | **LSTM**       | 注意力机制、DeepSHAP、序列可视化 | 捕捉时间序列模式，动态解释         | 解释复杂度高，需额外可视化支持     |
   | **GNN**        | 子图解释、SHAP、网络可视化      | 适合网络分析，揭示团伙行为         | 计算复杂，需图处理经验            |

9.  这种模型的解释性你有研究吗?怎么做反洗钱模型的解释性?
10. 这个模型输入是单条交易线输入吗?有没有办法输入他一条线这种连续的感觉的那种交易
11. GNN怎么做呢?里面
12. 用户画像,这种标签怎么输入呢?
13. 准确率多少,怎么修改,怎么优化
    ```
    - **准确率评估**：准确率受不平衡数据影响，需重点关注召回率和F1分数。
    - **修改方法**：
      - 数据：处理不平衡、清洗缺失值、增强标签质量。
      - 特征：优化用户画像标签，增加动态和网络特征。
      - 模型：调参、集成、引入注意力机制或子图采样。
    - **优化策略**：
      - 随机森林/XGBoost：超参数调优、规则提取。
      - LSTM：优化序列长度、加入注意力机制。
      - GNN：增强节点/边特征、剪枝优化。
    - **结合你的背景**：
      - 利用ETL自动化数据处理，实时更新用户画像。
      - 使用BI工具可视化性能和解释性，嵌入监管报送流程。

      见下方
   ```

14. 算法题目一道:
```
一系列字符串时，需要找到这些字符串中共有的最长前缀。
如果这些字符串之间不存在公共前缀，那么函数应该返回空字符串 ""
备注:
1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] 仅由小写英文字母组成
```

1.  你对这个收入的诉求大概是什么样子?
2.  工作对你来说是一种使命感还是一个工作?
3.  你这个简历里面这个论文讲一下




---

### 1. 用户画像标签的定义与作用
用户画像标签是描述客户特征的结构化数据，通常包括以下几类：
- **静态标签**：基于客户注册信息，例如年龄、职业、收入水平、账户类型、注册地等。
- **动态标签**：基于交易行为，例如交易频率、金额分布、账户活跃度、资金流向等。
- **风险标签**：基于外部数据或规则，例如高风险国家/地区关联、行业风险评分、制裁名单匹配。
- **行为模式标签**：基于历史交易模式，例如是否偏好夜间交易、是否经常与高风险账户交互。

**作用**：
- **提升模型精度**：用户画像为模型提供上下文信息，帮助区分正常交易和异常交易（例如，低收入客户的大额交易更可能是可疑的）。
- **支持合规性**：KYC信息是监管要求的核心，画像标签可直接用于监管报送（如中国人民银行的STR）。
- **增强解释性**：结合你的解释性需求，用户画像标签可以明确模型为何将某交易标记为可疑（例如，“客户职业为学生，交易金额异常高”）。

---

### 2. 用户画像标签的构建
用户画像标签的构建需要从多源数据中提取并整合特征，以下是常见类别和构建方法：

#### 2.1 静态标签
- **数据来源**：KYC数据、客户注册信息、外部数据库（如工商注册、信用报告）。
- **常见标签**：
  - **客户属性**：年龄、性别、职业（如学生、自由职业、企业主）、收入水平。
  - **账户信息**：账户类型（个人/企业）、开立时间、注册地。
  - **合规信息**：是否在制裁名单、是否涉及高风险国家/地区。
- **构建方法**：
  - 从核心银行系统或CRM数据库提取KYC数据。
  - 使用规则映射风险标签，例如：
    - 如果客户注册地为FATF灰名单国家，标记为“高风险”。
    - 如果职业为高风险行业（如博彩、虚拟货币），标记为“高风险行业”。

#### 2.2 动态标签
- **数据来源**：交易记录、账户流水。
- **常见标签**：
  - **交易行为**：月均交易金额、交易频率、夜间交易比例。
  - **资金流向**：与高风险账户的交易比例、跨境交易占比。
  - **行为变化**：交易金额/频率与历史均值的偏差。
- **构建方法**：
  - 使用时间窗口（例如，30天、90天）聚合交易数据，计算统计特征。
    - 示例：`avg_transaction_amount = sum(amount) / count(transactions)`。
    - 示例：`high_risk_interaction = count(transactions with high_risk_accounts) / total_transactions`。
  - 结合你的ETL经验，自动化计算动态标签（如使用Apache Spark处理大批量交易）。

#### 2.3 风险标签
- **数据来源**：外部风险数据库（如World-Check、Dow Jones）、监管名单、行业风险评分。
- **常见标签**：
  - 高风险国家/地区：基于FATF名单或监管要求。
  - 高风险行业：博彩、贵金属、虚拟货币等。
  - 制裁名单匹配：是否与制裁实体关联。
- **构建方法**：
  - 定期从外部API（如World-Check）更新风险名单。
  - 使用规则或机器学习模型为客户分配风险评分（例如，低/中/高风险）。

#### 2.4 行为模式标签
- **数据来源**：历史交易数据、时间序列分析、网络分析。
- **常见标签**：
  - 交易时间偏好：夜间交易占比、节假日交易占比。
  - 网络特征：账户在交易网络中的中心性（GNN计算）、与可疑账户的关联度。
  - 异常模式：交易金额突增、交易频率异常。
- **构建方法**：
  - 使用时间序列分析（如LSTM）提取交易模式。
  - 使用GNN分析账户在交易网络中的角色（如核心节点、桥接节点）。

**示例用户画像标签**：
| 标签类别       | 标签名称                     | 示例值                          | 数据来源           |
|----------------|-----------------------------|---------------------------------|-------------------|
| 静态标签       | age                         | 30                              | KYC数据           |
| 静态标签       | occupation                  | 学生                            | KYC数据           |
| 静态标签       | is_high_risk_country        | 1（是）                         | FATF名单          |
| 动态标签       | avg_transaction_amount      | 5000元                          | 交易数据          |
| 动态标签       | transaction_frequency       | 10次/月                        | 交易数据          |
| 风险标签       | risk_score                  | 高                              | 外部风险数据库     |
| 行为模式标签   | night_transaction_ratio     | 0.4（40%夜间交易）             | 交易时间分析      |
| 行为模式标签   | network_centrality         | 0.8（GNN计算的中心性）         | 交易网络分析      |

---

### 3. 用户画像标签的输入方式
用户画像标签可以作为模型的输入特征，具体输入方式取决于模型类型（单条交易模型或连续交易模型）。

#### 3.1 单条交易模型（随机森林/XGBoost）
- **输入方式**：
  - 将用户画像标签与单条交易特征合并，形成一个特征向量。
  - 示例：对于一笔交易，输入特征包括：
    - 交易特征：`amount`, `transaction_type`, `timestamp`。
    - 用户画像标签：`age`, `occupation`, `risk_score`, `avg_transaction_amount`。
- **处理步骤**：
  1. 从KYC和交易数据库提取用户画像标签。
  2. 将标签与交易记录通过`account_id`关联，生成完整特征集。
  3. 标准化/编码特征（例如，独热编码`occupation`，标准化`avg_transaction_amount`）。
  4. 输入模型进行预测。

**代码示例（单条交易输入，包含用户画像）**：
```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 加载交易和用户画像数据
transactions = pd.read_csv('transactions.csv')  # amount, transaction_type, account_id
profiles = pd.read_csv('user_profiles.csv')    # account_id, age, occupation, risk_score

# 合并数据
data = transactions.merge(profiles, on='account_id')

# 特征编码
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(data[['occupation', 'transaction_type']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# 合并编码后的特征
X = pd.concat([data[['amount', 'age', 'risk_score']], encoded_df], axis=1)
y = data['is_suspicious']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# 预测新交易
new_transaction = pd.DataFrame({
    'amount': [5000], 'age': [25], 'risk_score': [0.8],
    'occupation': ['student'], 'transaction_type': ['transfer']
})
new_encoded = encoder.transform(new_transaction[['occupation', 'transaction_type']])
new_encoded_df = pd.DataFrame(new_encoded, columns=encoder.get_feature_names_out())
new_input = pd.concat([new_transaction[['amount', 'age', 'risk_score']], new_encoded_df], axis=1)
prediction = model.predict(new_input)
print(f"Prediction: {'Suspicious' if prediction[0] == 1 else 'Normal'}")
```

#### 3.2 连续交易模型（LSTM）
- **输入方式**：
  - 将用户画像标签作为静态特征，与交易序列特征结合，形成混合输入。
  - 示例：每个交易序列（7天窗口）包含：
    - 动态特征：每日`amount`, `transaction_count`。
    - 静态特征：`age`, `risk_score`, `occupation`（对每个时间步重复）。
- **处理步骤**：
  1. 按账户和时间构造交易序列。
  2. 将用户画像标签附加到每个序列（作为全局特征或重复特征）。
  3. 输入LSTM模型，静态特征可通过全连接层与序列特征融合。

**代码示例（LSTM输入，包含用户画像）**：
```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Concatenate, Input
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 加载数据
transactions = pd.read_csv('transaction_sequence.csv')  # account_id, timestamp, amount, transaction_count
profiles = pd.read_csv('user_profiles.csv')             # account_id, age, occupation, risk_score

# 构造序列
def create_sequences(df, profiles, window_size=7):
    sequences, static_features, labels = [], [], []
    for account_id in df['account_id'].unique():
        account_data = df[df['account_id'] == account_id].sort_values('timestamp')
        profile_data = profiles[profiles['account_id'] == account_id][['age', 'risk_score']]
        for i in range(len(account_data) - window_size):
            seq = account_data[['amount', 'transaction_count']].iloc[i:i+window_size].values
            label = account_data['is_suspicious'].iloc[i+window_size-1]
            sequences.append(seq)
            static_features.append(profile_data.values)
            labels.append(label)
    return np.array(sequences), np.array(static_features), np.array(labels)

X_seq, X_static, y = create_sequences(transactions, profiles)

# 标准化
scaler_seq = StandardScaler()
X_seq = scaler_seq.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
scaler_static = StandardScaler()
X_static = scaler_static.fit_transform(X_static.reshape(-1, X_static.shape[-1])).reshape(X_static.shape)

# 构建LSTM模型（融合静态特征）
sequence_input = Input(shape=(X_seq.shape[1], X_seq.shape[2]), name='sequence')
static_input = Input(shape=(X_static.shape[2],), name='static')
lstm_out = LSTM(64)(sequence_input)
combined = Concatenate()([lstm_out, static_input])
dense = Dense(32, activation='relu')(combined)
output = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=[sequence_input, static_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([X_seq, X_static], y, epochs=10, batch_size=32)

# 预测新序列
new_sequence = np.array([[[5000, 2], [6000, 3], [4000, 1], [7000, 4], [3000, 2], [8000, 5], [9000, 3]]])
new_static = np.array([[25, 0.8]])  # age, risk_score
new_sequence = scaler_seq.transform(new_sequence.reshape(-1, new_sequence.shape[-1])).reshape(new_sequence.shape)
new_static = scaler_static.transform(new_static)
prediction = model.predict([new_sequence, new_static])
print(f"Prediction: {'Suspicious' if prediction[0] > 0.5 else 'Normal'}")
```

#### 3.3 交易网络模型（GNN）
- **输入方式**：
  - 将用户画像标签作为节点特征，结合交易特征（边特征）输入GNN。
  - 示例：节点特征包括`age`, `risk_score`, `avg_transaction_amount`，边特征包括`amount`, `timestamp`。
- **处理步骤**：
  1. 构建交易图，节点为账户，边为交易。
  2. 将用户画像标签附加到节点特征。
  3. 输入GNN模型，学习账户间的关系和可疑性。

**代码示例（GNN输入，包含用户画像）**：
```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd

# 加载数据
edges = pd.read_csv('transaction_edges.csv')  # source_account, target_account, amount
profiles = pd.read_csv('user_profiles.csv')   # account_id, age, risk_score, avg_transaction_amount

# 构造图数据
edge_index = torch.tensor(edges[['source_account', 'target_account']].values.T, dtype=torch.long)
edge_attr = torch.tensor(edges[['amount']].values, dtype=torch.float)
node_features = torch.tensor(profiles[['age', 'risk_score', 'avg_transaction_amount']].values, dtype=torch.float)
labels = torch.tensor(profiles['is_suspicious'].values, dtype=torch.long)
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)

# 构建GNN模型
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(node_features.shape[1], 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)

model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

# 预测
model.eval()
pred = model(data).argmax(dim=1)
print(f"Predicted suspicious accounts: {pred[pred == 1].tolist()}")
```

---

### 4. 结合你的背景的具体建议
基于你在监管报送、风险评估、ETL优化和BI工具的经验，以下是用户画像标签输入的具体建议：
1. **ETL流程优化**：
   - 设计自动化数据管道（如Apache Airflow、Spark），实时从KYC数据库、交易系统和外部风险数据库提取用户画像标签。
   - 示例：每天更新`risk_score`（基于最新制裁名单），每月更新`avg_transaction_amount`。
2. **特征工程**：
   - 静态标签：对类别特征（如`occupation`）进行独热编码，对数值特征（如`age`）标准化。
   - 动态标签：使用滑动窗口计算行为特征（如30天平均交易金额），结合你的ETL经验自动化生成。
   - 风险标签：定期同步外部数据（如FATF名单），通过规则或模型生成风险评分。
3. **模型融合**：
   - 单条交易模型：将用户画像标签直接并入特征向量，适合快速检测。
   - LSTM：将静态画像标签与序列特征融合，捕捉行为与背景的综合影响。
   - GNN：将画像标签作为节点特征，分析账户网络中的风险角色。
4. **解释性增强**：
   - 使用SHAP分析用户画像标签的贡献度，例如“高风险国家标签增加0.3可疑概率”。
   - 结合BI工具（如Tableau）可视化画像标签与交易特征的关系，生成监管报告。
5. **合规性保障**：
   - 确保用户画像标签符合《个人信息保护法》，对敏感信息（如姓名）进行脱敏。
   - 在监管报送中，将画像标签（如`risk_score`）与交易上下文结合，生成清晰的STR说明。

---

### 5. 用户画像标签输入的注意事项
- **数据质量**：确保KYC数据完整，避免缺失值影响模型性能。
- **特征选择**：通过特征重要性分析（如SHAP）筛选高贡献标签，减少冗余。
- **实时性**：结合你的ETL经验，构建实时更新用户画像的管道，适应动态变化（如客户收入变化）。
- **隐私合规**：对用户画像中的敏感信息进行加密或匿名化处理。
- **不平衡数据**：高风险客户通常占少数，使用加权损失或过采样技术平衡数据集。

---

### 6. 总结
- **用户画像标签的构建**：
  - 静态标签：从KYC数据提取客户属性（如年龄、职业）。
  - 动态标签：从交易数据计算行为特征（如平均交易金额）。
  - 风险标签：结合外部数据库生成风险评分。
  - 行为模式标签：通过时间序列或网络分析提取模式。
- **输入方式**：
  - **单条交易模型**：将画像标签与交易特征合并为特征向量。
  - **LSTM**：将静态画像标签与交易序列融合，输入混合模型。
  - **GNN**：将画像标签作为节点特征，结合交易边特征。
- **结合你的背景**：
  - 利用ETL自动化数据管道，实时更新用户画像。
  - 使用BI工具可视化标签贡献，增强解释性。
  - 确保标签符合监管要求，嵌入STR报送流程。





---

### 1. 模型准确率的定义与评估
#### 1.1 准确率的定义
在AML模型中，准确率（Accuracy）通常定义为：
\[
\text{Accuracy} = \frac{\text{正确预测的样本数}}{\text{总样本数}} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
\]
- TP（True Positive）：正确预测为可疑交易的样本。
- TN（True Negative）：正确预测为正常交易的样本。
- FP（False Positive）：错误预测为可疑交易的正常交易。
- FN（False Negative）：错误预测为正常交易的可疑交易。

#### 1.2 反洗钱场景中的准确率
- **典型准确率**：在AML场景中，由于可疑交易（正样本）通常远少于正常交易（负样本），数据高度不平衡，准确率可能很高（如95%+），但这可能掩盖模型对可疑交易的低召回率。
- **更重要的指标**：
  - **召回率（Recall）**：\(\frac{\text{TP}}{\text{TP} + \text{FN}}\)，表示检测到所有可疑交易的能力，AML中优先级高，因为漏检（FN）成本高。
  - **精确率（Precision）**：\(\frac{\text{TP}}{\text{TP} + \text{FP}}\)，表示预测为可疑的交易中正确比例，影响人工审核成本。
  - **F1分数**：\(\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}\)，平衡精确率和召回率。
  - **ROC-AUC**：衡量模型区分正负样本的能力，适合不平衡数据。
- **实际案例**：根据公开研究和行业实践，AML模型的召回率通常在60%-90%（取决于数据集和模型），精确率可能较低（20%-50%），因为可疑交易稀少，FP较多。

#### 1.3 评估方法
- **交叉验证**：使用k折交叉验证（如5折）评估模型稳定性。
- **混淆矩阵**：分析TP、TN、FP、FN的分布，识别模型偏见。
- **业务指标**：结合你的监管报送经验，评估模型是否满足监管要求（如减少漏报，确保STR质量）。

**代码示例（评估准确率等指标）**：
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv('transactions.csv')
X = data.drop(columns=['is_suspicious'])
y = data['is_suspicious']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估指标
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

### 2. 影响准确率的因素
以下是可能导致AML模型准确率（或召回率、精确率）不理想的常见原因：
1. **数据质量问题**：
   - 缺失值：KYC数据不完整（如职业、收入缺失）。
   - 标签噪声：历史数据中可疑交易标签不准确（人工标注错误）。
   - 不平衡数据：可疑交易比例低（例如，1%或更低）。
2. **特征工程不足**：
   - 用户画像标签单一，未能捕捉客户行为全貌。
   - 交易序列或网络特征未充分利用，忽略连续性或关联性。
3. **模型选择与调参**：
   - 随机森林/XGBoost：树深度不足或过拟合。
   - LSTM：序列长度不合适，隐藏层设置不佳。
   - GNN：图结构复杂，节点/边特征不充分。
4. **业务场景复杂性**：
   - 洗钱手法多样（如分拆交易、团伙洗钱），单一模型难以覆盖。
   - 监管规则变化，模型未及时更新。
5. **实时性与延迟**：ETL流程未优化，导致输入数据滞后或不完整。

---

### 3. 修改与优化方法
以下针对随机森林、LSTM、GNN模型，以及用户画像标签的输入，提供具体的修改和优化策略。

#### 3.1 数据优化
- **处理不平衡数据**：
  - **过采样/欠采样**：使用SMOTE（Synthetic Minority Oversampling Technique）生成可疑交易样本，或随机欠采样正常交易。
  - **加权损失**：在模型中设置正样本权重（如XGBoost的`scale_pos_weight`）。
  - **代码示例**：
    ```python
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    model.fit(X_resampled, y_resampled)
    ```
- **数据清洗**：
  - 填充缺失值：使用均值/中位数填充数值特征，众数填充类别特征。
  - 异常值处理：剔除或截断异常交易金额（如金额>99.9%分位数）。
  - 结合你的ETL经验：使用Spark或Airflow自动化清洗流程。
- **标签质量**：
  - 人工复核历史标签，确保可疑交易标注准确。
  - 引入外部数据（如FATF名单、World-Check）增强标签可靠性。

#### 3.2 特征工程优化
- **增强用户画像标签**：
  - **静态标签**：增加更多KYC特征，如客户信用评分、账户用途（投资/消费）。
  - **动态标签**：计算更复杂的交易特征，如交易金额的滑动窗口标准差、与高风险账户的交互频率。
  - **网络标签**：使用GNN提取账户网络特征（如中心性、社区结构），加入用户画像。
  - **代码示例（动态标签）**：
    ```python
    import pandas as pd
    # 计算30天滑动窗口特征
    transactions['amount_std'] = transactions.groupby('account_id')['amount'].rolling(30).std().reset_index(drop=True)
    transactions['high_risk_interaction'] = transactions.groupby('account_id').apply(
        lambda x: (x['is_high_risk_account'] * x['amount']).sum() / x['amount'].sum()
    ).reset_index(drop=True)
    ```
- **时间序列特征（LSTM）**：
  - 优化窗口大小：测试不同时间窗口（7天、14天、30天），选择召回率最高的。
  - 增加序列统计：如序列的傅里叶变换（捕捉周期性）、波动率。
- **网络特征（GNN）**：
  - 提取更多边特征：如交易时间间隔、资金流转速度。
  - 使用社区检测算法（如Louvain）识别潜在洗钱团伙。

#### 3.3 模型优化
- **随机森林/XGBoost**：
  - **超参数调优**：使用网格搜索或贝叶斯优化调整`max_depth`, `learning_rate`, `n_estimators`。
  - **集成模型**：结合多棵树或不同模型（如XGBoost+LightGBM）提升性能。
  - **代码示例（网格搜索）**：
    ```python
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200]
    }
    grid_search = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    print("Best Params:", grid_search.best_params_)
    ```
- **LSTM**：
  - **网络结构**：增加隐藏层或神经元（如64→128），加入Dropout防止过拟合。
  - **注意力机制**：增强模型对关键时间点的关注，提升召回率。
  - **序列长度**：测试不同时间步（如5、10、15），平衡计算复杂度和性能。
- **GNN**：
  - **模型选择**：尝试GraphSAGE、GAT（图注意力网络），提升节点嵌入质量。
  - **图剪枝**：移除低权重边，减少计算复杂性。
  - **子图采样**：对大规模交易网络，使用子图采样（如Cluster-GCN）加速训练。
- **混合模型**：
  - 结合随机森林（单条交易）、LSTM（序列）和GNN（网络），通过Stacking或加权投票集成。
  - 示例：用随机森林初步筛选可疑交易，LSTM验证序列模式，GNN确认团伙行为。

#### 3.4 解释性优化
- **SHAP/LIME**：分析特征贡献，剔除低重要性特征，聚焦高贡献用户画像标签（如`risk_score`）。
- **规则提取**：从树模型中提取规则（如“金额>10,000且risk_score>0.8”），验证是否提升召回率。
- **可视化**：结合你的BI工具经验，用Tableau展示特征重要性、交易序列或网络，辅助业务人员优化模型。

#### 3.5 实时性与自动化
- **ETL优化**：利用你的ETL经验，构建实时数据管道（如Kafka+Spark），确保用户画像标签和交易数据及时更新。
- **模型部署**：使用Docker或Kubernetes部署模型，支持实时推理。
- **增量学习**：定期用新数据更新模型（如每周重新训练），适应洗钱手法变化。

---

### 4. 结合你的背景的具体优化建议
1. **监管报送需求**：
   - 优化召回率优先：调整模型损失函数，增加正样本权重，确保不漏检可疑交易。
   - 生成解释性报告：将SHAP值和用户画像标签（如`high_risk_country`）嵌入STR，满足人民银行要求。
2. **ETL与数据管道**：
   - 自动化用户画像更新：每天从KYC和外部风险数据库（如World-Check）更新`risk_score`等标签。
   - 实时特征计算：使用Spark Streaming计算动态标签（如`avg_transaction_amount`）。
3. **BI工具可视化**：
   - 绘制混淆矩阵和ROC曲线，分析模型性能。
   - 可视化用户画像标签贡献（如`age`对可疑交易的SHAP值分布）。
4. **模型选择与融合**：
   - 结合随机森林（快速筛选）、LSTM（序列模式）、GNN（团伙检测），提升综合性能。
   - 使用Stacking集成：随机森林输出初步概率，LSTM和GNN进行二次验证。
5. **合规性保障**：
   - 确保用户画像标签符合《个人信息保护法》，对敏感特征（如客户姓名）脱敏。
   - 定期验证模型输出是否满足FATF或人民银行的监管标准。

### 6. 注意事项
- **过拟合风险**：监控训练集和测试集性能差距，使用正则化（如Dropout、L2正则）防止过拟合。
- **数据隐私**：确保用户画像标签和交易数据符合《数据安全法》，避免敏感信息泄露。
- **监管动态**：定期更新高风险国家/行业名单，调整用户画像标签。
- **人工复核**：结合你的监管报送经验，确保模型输出易于人工审核（如提供SHAP解释）。

---

### 7. 总结
- **准确率评估**：准确率受不平衡数据影响，需重点关注召回率和F1分数。
- **修改方法**：
  - 数据：处理不平衡、清洗缺失值、增强标签质量。
  - 特征：优化用户画像标签，增加动态和网络特征。
  - 模型：调参、集成、引入注意力机制或子图采样。
- **优化策略**：
  - 随机森林/XGBoost：超参数调优、规则提取。
  - LSTM：优化序列长度、加入注意力机制。
  - GNN：增强节点/边特征、剪枝优化。
- **结合你的背景**：
  - 利用ETL自动化数据处理，实时更新用户画像。
  - 使用BI工具可视化性能和解释性，嵌入监管报送流程。
