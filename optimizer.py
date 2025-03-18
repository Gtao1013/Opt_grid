import numpy as np
from typing import List, Tuple, Optional, Union


class NSGAIIParameters:
    """NSGA-II算法的参数类"""

    def __init__(self, pop_size=100, crossover_prob=0.9, crossover_eta=20,
                 mutation_prob=None, mutation_eta=20):
        self.pop_size = int(pop_size)  # 确保是整数
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob
        self.mutation_eta = mutation_eta

    def get_pop_size(self):
        return self.pop_size


class NSGA2Optimizer:
    """NSGA-II多目标优化算法实现"""

    def __init__(self, n_var: int, n_obj: int, bounds: np.ndarray,
                 params: Optional[Union[NSGAIIParameters, dict]] = None,
                 minimize: List[bool] = None) -> None:
        """
        初始化NSGA-II优化器

        Parameters:
        -----------
        n_var : int
            决策变量数量
        n_obj : int
            目标函数数量
        bounds : np.ndarray
            决策变量的边界约束，形状为(n_var, 2)
        params : NSGAIIParameters or dict, optional
            算法参数，默认为None
        minimize : List[bool], optional
            每个目标是否为最小化，默认为None（表示全部最小化）
        """
        self.n_var = n_var
        self.n_obj = n_obj
        self.bounds = np.array(bounds)

        # 设置最小化/最大化标志
        if minimize is None:
            self.minimize = [True] * n_obj  # 默认所有目标都是最小化
        else:
            self.minimize = minimize

        # 处理参数
        if params is None:
            params = NSGAIIParameters()
        elif isinstance(params, dict):
            params = NSGAIIParameters(**params)

        self.params = params
        self.pop_size = params.get_pop_size()

        # 初始化种群和目标值
        self.population = None
        self.objectives = None

    def initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.zeros((self.pop_size, self.n_var))
        for i in range(self.n_var):
            population[:, i] = np.random.uniform(
                self.bounds[i, 0],
                self.bounds[i, 1],
                size=self.pop_size
            )
        return population

    def simulated_binary_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """模拟二进制交叉"""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                # SBX交叉
                if np.random.random() < self.params.crossover_prob:
                    beta = np.random.random()
                    if beta <= 0.5:
                        beta = (2 * beta) ** (1 / (self.params.crossover_eta + 1))
                    else:
                        beta = (1 / (2 * (1 - beta))) ** (1 / (self.params.crossover_eta + 1))

                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

                    # 边界处理
                    child1[i] = np.clip(child1[i], self.bounds[i, 0], self.bounds[i, 1])
                    child2[i] = np.clip(child2[i], self.bounds[i, 0], self.bounds[i, 1])
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]

        return child1, child2

    def polynomial_mutation(self, individual: np.ndarray) -> None:
        """多项式变异"""
        for i in range(len(individual)):
            if np.random.random() < self.params.mutation_prob:
                delta1 = (individual[i] - self.bounds[i, 0]) / (self.bounds[i, 1] - self.bounds[i, 0])
                delta2 = (self.bounds[i, 1] - individual[i]) / (self.bounds[i, 1] - self.bounds[i, 0])

                rand = np.random.random()
                mut_pow = 1.0 / (self.params.mutation_eta + 1.0)

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.params.mutation_eta + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.params.mutation_eta + 1.0))
                    delta_q = 1.0 - val ** mut_pow

                individual[i] = individual[i] + delta_q * (self.bounds[i, 1] - self.bounds[i, 0])
                individual[i] = np.clip(individual[i], self.bounds[i, 0], self.bounds[i, 1])

    def create_offspring(self, population: np.ndarray) -> np.ndarray:
        """创建子代"""
        offspring = np.zeros_like(population)
        shuffle_indices = np.random.permutation(len(population))

        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                offspring[i], offspring[i + 1] = self.simulated_binary_crossover(
                    population[shuffle_indices[i]],
                    population[shuffle_indices[i + 1]]
                )

        for i in range(len(offspring)):
            self.polynomial_mutation(offspring[i])

        return offspring

    def calculate_crowding_distance(self, objectives):
        """
        计算拥挤度距离

        Parameters:
        -----------
        objectives : np.ndarray
            目标函数值数组

        Returns:
        --------
        np.ndarray
            拥挤度距离数组
        """
        n_points = len(objectives)
        if n_points <= 2:
            return np.full(n_points, np.inf)

        distances = np.zeros(n_points)

        # 对每个目标计算拥挤度
        for i in range(self.n_obj):
            # 获取该目标的排序索引
            sorted_indices = np.argsort(objectives[:, i])

            # 设置边界点的拥挤度为无穷大
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # 计算中间点的拥挤度
            obj_range = objectives[sorted_indices[-1], i] - objectives[sorted_indices[0], i]
            if obj_range > 0:
                for j in range(1, n_points - 1):
                    distances[sorted_indices[j]] += (
                                                            objectives[sorted_indices[j + 1], i] -
                                                            objectives[sorted_indices[j - 1], i]
                                                    ) / obj_range

        return distances

    def fast_non_dominated_sort(self, objectives):
        """执行快速非支配排序，适应最大化优化问题"""
        n_points = len(objectives)

        # 初始化支配计数和被支配解的列表
        domination_count = np.zeros(n_points, dtype=int)
        dominated_solutions = [[] for _ in range(n_points)]
        fronts = [[]]  # 第一个前沿

        # 计算支配关系
        for i in range(n_points):
            for j in range(i + 1, n_points):
                i_dominates = False
                j_dominates = False

                # 检查每个目标的支配关系，考虑最大化/最小化标志
                for k in range(self.n_obj):
                    # 根据目标是最小化还是最大化确定比较方式
                    if self.minimize[k]:
                        # 最小化目标，值越小越好
                        if objectives[i][k] < objectives[j][k]:
                            i_dominates = True
                        elif objectives[i][k] > objectives[j][k]:
                            j_dominates = True
                    else:
                        # 最大化目标，值越大越好
                        if objectives[i][k] > objectives[j][k]:  # 注意这里是 > 而不是 <
                            i_dominates = True
                        elif objectives[i][k] < objectives[j][k]:  # 注意这里是 < 而不是 >
                            j_dominates = True

                    # 如果已经确定了两个解互相支配，则不需继续比较
                    if i_dominates and j_dominates:
                        break

                # 更新支配关系
                if i_dominates and not j_dominates:
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif j_dominates and not i_dominates:
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # 找出第一个非支配前沿
        for i in range(n_points):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # 构建剩余的前沿
        i = 0
        while i < len(fronts):
            current = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        current.append(q)
            i += 1
            if current:
                fronts.append(current)

        return fronts


    def run_generation(self, evaluate_func):
        """运行一代优化"""
        try:
            # 初始化种群（如果是第一代）
            if self.population is None:
                self.population = self.initialize_population()
                self.objectives = np.array([evaluate_func(ind) for ind in self.population])

            # 生成子代
            offspring = self.create_offspring(self.population)
            offspring_obj = np.array([evaluate_func(ind) for ind in offspring])

            # 合并父代和子代
            combined_pop = np.vstack((self.population, offspring))
            combined_obj = np.vstack((self.objectives, offspring_obj))

            # 非支配排序
            fronts = self.fast_non_dominated_sort(combined_obj)

            # 选择下一代
            next_pop = []
            next_obj = []
            front_no = 0

            # 按前沿顺序添加解
            while len(next_pop) + len(fronts[front_no]) <= self.pop_size:
                for idx in fronts[front_no]:
                    next_pop.append(combined_pop[idx])
                    next_obj.append(combined_obj[idx])
                front_no += 1
                if front_no >= len(fronts):
                    break

            # 如果最后一个前沿没有完全加入，使用拥挤度排序
            if len(next_pop) < self.pop_size and front_no < len(fronts):
                # 计算最后一个前沿的拥挤度
                last_front = fronts[front_no]
                crowding_distances = self.calculate_crowding_distance(combined_obj[last_front])

                # 根据拥挤度排序
                sorted_indices = np.argsort(crowding_distances)[::-1]
                remaining = self.pop_size - len(next_pop)

                # 添加剩余的解
                for idx in sorted_indices[:remaining]:
                    next_pop.append(combined_pop[last_front[idx]])
                    next_obj.append(combined_obj[last_front[idx]])

            # 更新种群
            self.population = np.array(next_pop)
            self.objectives = np.array(next_obj)

            # 打印当前代的统计信息
            print("\n当前种群统计:")
            print(f"种群大小: {len(self.population)}")
            print(f"目标函数范围:")
            for i, name in enumerate(["Cl", "Cd", "Cm"]):
                obj_vals = self.objectives[:, i]
                print(f"{name}: [{obj_vals.min():.4f}, {obj_vals.max():.4f}]")

            return self.population, self.objectives

        except Exception as e:
            print(f"\n运行代出现错误: {str(e)}")
            raise

