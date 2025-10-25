from env import CodeOptimizationEnv
from agent import CodeOptimizationAgent
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import random

if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer
    export CUDA_VISIBLE_DEVICES=2
    python train.py

    pip install stable-baselines3 gym shimmy astunparse
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 使用 function_corpus.json 文件
    function_corpus_file = "function_corpus.json"
    env = make_vec_env(lambda: CodeOptimizationEnv(function_corpus_file), n_envs=1)
    env.seed(42)

    # 初始化并训练代理
    agent = CodeOptimizationAgent(env)
    best_reward = float("-inf")

    # 训练循环
    for _ in range(10000 // 2048):
        agent.train(total_timesteps=2048)
        obs = env.reset()
        total_reward = 0
        for _ in range(10):
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        if total_reward > best_reward:
            best_reward = total_reward
            agent.model.save("ppo_code_optimizer")
            print(f"Saved model with total reward: {total_reward}")

    # 测试所有函数的优化结果
    env = CodeOptimizationEnv(function_corpus_file)
    for func_idx in range(len(env.function_corpus)):
        env.current_func_idx = func_idx
        obs = env.reset()
        print(f"\nOptimizing function {func_idx + 1}:")
        print(f"Initial Code:\n{env.current_code}")
        for i in range(10):
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            print(f"Step {i+1}:")
            print(f"Code:\n{env.current_code}")
            print(f"Reward: {reward}")
            print("-" * 50)
            if done:
                print("Optimization stopped.")
                break
