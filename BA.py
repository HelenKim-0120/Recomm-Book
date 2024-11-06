import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# 初始完全图生成
def create_complete_graph(n):
    graph = {i: set(range(i)) | set(range(i + 1, n)) for i in range(n)}
    return graph

# 按照BA模型方式添加新节点的函数
def add_node_with_preferential_attachment(graph, c):
    new_node = len(graph)
    node_degrees = [(node, len(neighbors)) for node, neighbors in graph.items()]
    total_degree = sum(degree for node, degree in node_degrees)
    
    targets = set()
    while len(targets) < c:
        r = random.uniform(0, total_degree)
        cumulative_sum = 0
        for node, degree in node_degrees:
            cumulative_sum += degree
            if cumulative_sum >= r and node not in targets:
                targets.add(node)
                break
    
    graph[new_node] = targets
    for target in targets:
        graph[target].add(new_node)

# 调查度分布的函数
def degree_distribution(graph):
    degree_freq = {}
    for node, neighbors in graph.items():
        degree = len(neighbors)
        if degree not in degree_freq:
            degree_freq[degree] = 0
        degree_freq[degree] += 1
    return degree_freq

# 调查聚类系数的函数
def clustering_coefficient(graph):
    clustering = []
    for node, neighbors in graph.items():
        if len(neighbors) < 2:
            continue
        links = 0
        for neighbor in neighbors:
            links += len(neighbors & graph[neighbor])
        clustering.append(links / (len(neighbors) * (len(neighbors) - 1)))
    return np.mean(clustering)

# 调查直径的函数
def network_diameter(graph):
    def bfs_longest_path(start_node):
        visited = set()
        queue = deque([(start_node, 0)])
        max_distance = 0
        while queue:
            current_node, distance = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                max_distance = max(max_distance, distance)
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
        return max_distance

    max_diameter = 0
    for node in graph:
        max_diameter = max(max_diameter, bfs_longest_path(node))
    return max_diameter

# 使用最大似然估计法来估计幂律分布指数的函数
def estimate_power_law_exponent(degree_freq):
    degrees = []
    for degree, count in degree_freq.items():
        degrees.extend([degree] * count)
    
    degrees = np.array(degrees)
    degrees = degrees[degrees > 0]  # 去除度为0的节点
    xmin = min(degrees)
    return 1 + len(degrees) / np.sum(np.log(degrees / xmin))

# 可视化网络的函数
def visualize_graph(graph, title):
    plt.figure(figsize=(10, 8))
    pos = {node: (random.uniform(0, 1), random.uniform(0, 1)) for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            x_values = [pos[node][0], pos[neighbor][0]]
            y_values = [pos[node][1], pos[neighbor][1]]
            plt.plot(x_values, y_values, color='gray', alpha=0.5)
    
    for node in graph:
        plt.scatter(pos[node][0], pos[node][1], s=10, color='blue')
    
    plt.title(title)
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.show()

# 主要变量设置及尝试
n_factorial = 10000  # 初始完全图的节点数 (n!)
c_values = [3, 5]  # 要连接的边数

for c in c_values:
    print(f"\n\n### 尝试: c = {c} ###")
    graph = create_complete_graph(n_factorial)
    
    # 通过添加节点扩展为BA模型
    for _ in range(100):
        add_node_with_preferential_attachment(graph, c)
    
    # 可视化网络
    visualize_graph(graph, f'BA模型网络 (c={c})')
    
    # 调查度分布
    degree_freq = degree_distribution(graph)
    exponent = estimate_power_law_exponent(degree_freq)
    print("度分布 (幂律指数):", exponent)

    # 调查聚类系数
    clustering = clustering_coefficient(graph)
    print("聚类系数:", clustering)

    # 调查直径
    diameter = network_diameter(graph)
    print("直径:", diameter)

    # 结果可视化
    plt.bar(degree_freq.keys(), degree_freq.values(), color='blue', alpha=0.7)
    plt.xlabel('度')
    plt.ylabel('频率')
    plt.title(f'度分布 (c={c})')
    plt.show()