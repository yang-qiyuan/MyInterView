import gym
import ast
import numpy as np
import astunparse
from evaluate import evaluate_code
import json


class CodeOptimizationEnv(gym.Env):
    def __init__(self, function_corpus_file: str):
        super().__init__()
        # 加载 function_corpus.json
        with open(function_corpus_file, "r", encoding="utf-8") as file:
            self.function_corpus = json.load(file)["functions"]

        # 随机选择一个函数作为初始代码
        self.current_func_idx = 0
        self.initial_code = self.function_corpus[self.current_func_idx]["code"]
        self.current_code = self.initial_code
        self.test_cases = self.function_corpus[self.current_func_idx]["test_cases"]

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.max_steps = 50
        self.current_step = 0
        self.num_functions = len(self.function_corpus)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # 每次 reset 时随机选择一个函数
        self.current_func_idx = self.np_random.integers(0, self.num_functions)
        self.initial_code = self.function_corpus[self.current_func_idx]["code"]
        self.current_code = self.initial_code
        self.test_cases = self.function_corpus[self.current_func_idx]["test_cases"]
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        """获取当前状态"""
        try:
            tree = ast.parse(self.current_code)
            node_count = sum(1 for _ in ast.walk(tree))
            depth = max(
                (len(list(ast.iter_child_nodes(n))) for n in ast.walk(tree)), default=0
            )
            return np.array(
                [node_count / 100, depth / 10, len(self.current_code.split()) / 100]
                + [0] * 7
            )
        except SyntaxError:
            return np.zeros(10)

    def step(self, action: int):
        """执行一步动作，返回当前状态"""
        self.current_code = self._apply_transformation(action)
        reward = evaluate_code(self.current_code, self.test_cases)
        self.current_step += 1
        done = self.current_step >= self.max_steps or reward <= -5.0
        return self._get_state(), reward, done, {}

    def _apply_transformation(self, action: int) -> str:
        try:
            tree = ast.parse(self.current_code)
            if action == 0:  # 将 for 循环转换为列表推导式
                for node in ast.walk(tree):
                    if isinstance(node, ast.For):
                        if (
                            isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Call)
                            and isinstance(node.body[0].value.func, ast.Attribute)
                            and node.body[0].value.func.attr == "append"
                        ):
                            new_listcomp = ast.ListComp(
                                elt=node.body[0].value.args[0],
                                generators=[
                                    ast.comprehension(
                                        target=node.target, iter=node.iter, ifs=[]
                                    )
                                ],
                            )
                            assign = ast.Assign(
                                targets=[ast.Name(id="result", ctx=ast.Store())],
                                value=new_listcomp,
                            )
                            # 替换原始 for 循环
                            for i, stmt in enumerate(tree.body):
                                if stmt == node:
                                    tree.body[i] = assign
                                    break
                            break

            elif action == 1:  # 合并嵌套循环为单层循环
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.For)
                        and len(node.body) == 1
                        and isinstance(node.body[0], ast.For)
                    ):
                        inner_loop = node.body[0]
                        # 假设内外循环可以合并为单层循环（例如，生成笛卡尔积）
                        new_iter = ast.Call(
                            func=ast.Name(id="itertools.product", ctx=ast.Load()),
                            args=[node.iter, inner_loop.iter],
                            keywords=[],
                        )
                        new_target = ast.Tuple(
                            elts=[node.target, inner_loop.target], ctx=ast.Store()
                        )
                        new_comprehension = ast.comprehension(
                            target=new_target, iter=new_iter, ifs=[]
                        )
                        new_listcomp = ast.ListComp(
                            elt=(
                                inner_loop.body[0].value.args[0]
                                if isinstance(inner_loop.body[0], ast.Expr)
                                and isinstance(inner_loop.body[0].value, ast.Call)
                                else inner_loop.body[0]
                            ),
                            generators=[new_comprehension],
                        )
                        assign = ast.Assign(
                            targets=[ast.Name(id="result", ctx=ast.Store())],
                            value=new_listcomp,
                        )
                        for i, stmt in enumerate(tree.body):
                            if stmt == node:
                                tree.body[i] = assign
                                break
                        break

            elif action == 2:  # 替换低效的 sum([x for x in lst]) 为 sum(lst)
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "sum"
                        and len(node.args) == 1
                        and isinstance(node.args[0], ast.ListComp)
                    ):
                        new_call = ast.Call(
                            func=ast.Name(id="sum", ctx=ast.Load()),
                            args=[node.args[0].generators[0].iter],
                            keywords=[],
                        )
                        for parent in ast.walk(tree):
                            for attr, value in ast.iter_fields(parent):
                                if isinstance(value, list):
                                    for i, item in enumerate(value):
                                        if item == node:
                                            value[i] = new_call
                                            break
                        break

            elif action == 3:  # 合并多重 if 语句
                for node in ast.walk(tree):
                    if isinstance(node, ast.If):
                        if isinstance(node.body[0], ast.If):
                            # 合并嵌套 if 条件
                            combined_test = ast.BoolOp(
                                op=ast.And(),
                                values=[node.test, node.body[0].test],
                            )
                            new_if = ast.If(
                                test=combined_test,
                                body=node.body[0].body,
                                orelse=node.orelse or node.body[0].orelse,
                            )
                            for i, stmt in enumerate(tree.body):
                                if stmt == node:
                                    tree.body[i] = new_if
                                    break
                            break

            elif action == 4:  # 添加单行注释
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # 在函数定义后添加单行注释
                        comment = ast.Expr(
                            value=ast.Str(s=f"# Optimized function {node.name}")
                        )
                        for i, stmt in enumerate(tree.body):
                            if stmt == node:
                                tree.body.insert(i + 1, comment)
                                break
                        break

            return astunparse.unparse(tree).strip()
        except Exception:
            return self.current_code
