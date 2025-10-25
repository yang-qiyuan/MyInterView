20250630

老子等了20分钟,这人都不来,直接走了,妈的

20240703

1. 找个你觉得价值比较大的一个项目进行细讲
2. lora原理
3. 贝尔曼公式介绍
4. 强化学习策略评估 方法




计算给定策略下的状态价值（state value）

```python
policy = [
    ["→+0", "→+0", "→+0", "↓+0", "↓+0"],
    ["↑+0", "↑+0", "→+0", "↓+0", "↓+0"],
    ["↑+0", "←+0", "↓+1", "→+0", "↓+0"],
    ["↑+0", "→+1", "o+1", "↑+1", "↓+0"],
    ["↑+0", "→+0", "↑+1", "←+0", "←+0"]
]
```

1. **策略评估逻辑**：
   - 遍历每个状态 `(i, j)`，根据策略矩阵获取当前状态的动作和即时奖励。
   - 根据动作计算新状态的坐标，并处理边界条件（防止越界）。
   - 使用贝尔曼方程更新状态价值：$ V(s) = R(s, a) + \gamma V(s') $

2. **收敛条件**：
   - 使用 `delta` 跟踪每次迭代的状态价值变化。
   - 如果 `delta` 小于阈值 `theta`，则认为已经收敛，停止迭代。

```python
import numpy as np

# 定义策略矩阵
policy = [
    ["→+0", "→+0", "→+0", "↓+0", "↓+0"],
    ["↑+0", "↑+0", "→+0", "↓+0", "↓+0"],
    ["↑+0", "←+0", "↓+1", "→+0", "↓+0"],
    ["↑+0", "→+1", "0+1", "↑+1", "↓+0"],
    ["↑+0", "→+0", "↑+1", "←+0", "←+0"]
]

# 定义网格大小
grid_size = 5

# 定义折扣因子
gamma = 0.9

# 初始化状态价值矩阵
V = np.zeros((grid_size, grid_size))

# 定义动作映射
action_map = {
    "↑": (-1, 0),  # 向上
    "↓": (1, 0),   # 向下
    "←": (0, -1),  # 向左
    "→": (0, 1),   # 向右
    "o": (0, 0)    # 原地不动
}

# 策略评估函数
def policy_evaluation(V, policy, gamma, max_iterations=1000, theta=1e-6):
    for _ in range(max_iterations):
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                action_str = policy[i][j]
                direction, reward = action_str[:-2], int(action_str[-1])
                
                # 获取动作对应的移动方向
                di, dj = action_map[direction]
                
                # 计算新的状态坐标
                new_i, new_j = i + di, j + dj
                
                # 处理边界条件
                if new_i < 0:
                    new_i = 0
                elif new_i >= grid_size:
                    new_i = grid_size - 1
                if new_j < 0:
                    new_j = 0
                elif new_j >= grid_size:
                    new_j = grid_size - 1
                
                # 更新状态价值
                new_value = reward + gamma * V[new_i, new_j]
                delta = max(delta, abs(new_value - V[i, j]))
                V[i, j] = new_value
        
        # 检查收敛条件
        if delta < theta:
            break
    return V

# 进行策略评估
V = policy_evaluation(V, policy, gamma)

# 打印结果
print("State Values:")
print(V)
```
