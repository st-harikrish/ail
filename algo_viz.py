import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Graph:
    def __init__(self):
        self.graph = {}
        self.weight = {}
        self.heuristic = {}

    def addEdge(self, start, dest, weight=1):
        if start not in self.graph:
            self.graph[start] = []
            self.weight[start] = []
            self.heuristic[start] = 10

        if dest not in self.graph:
            self.graph[dest] = []
            self.weight[dest] = []
            self.heuristic[dest] = 10

        self.graph[start].append(dest)
        self.weight[start].append(weight)

        combined = sorted(zip(self.graph[start], self.weight[start]), key=lambda x: x[0])
        # print(combined)
        self.graph[start], self.weight[start] = map(list, zip(*combined))
        self.graph[dest].append(start)
        self.weight[dest].append(weight)
        combined = sorted(zip(self.graph[dest], self.weight[dest]), key=lambda x: x[0])
        self.graph[dest], self.weight[dest] = map(list, zip(*combined))

    def add_heuristics(self, start, h_val):
        self.heuristic[start] = h_val

    def __str__(self):
        return f"{self.graph}\n{self.weight}\n{self.heuristic}"

class Algorithm:
    # def BMS(self, graph, start, dest):
    #     paths = []
    #     stack = [(start, [start])]
    #     while stack:
    #         node, path = stack.pop()
    #         paths.append(path)
    #         for neighbor in graph.graph[node]:
    #             if neighbor not in path:
    #                 stack.append((neighbor, path + [neighbor]))
    #     print("BMS: ", paths)
    #     return paths

    def BMS(self, graph, start, dest):
        visited = set()
        queue = [(start, [start])]
        all_paths = []
        front = 0  # Index of front of the queue
        while front < len(queue):
            node, path = queue[front]
            all_paths.append(path)
            if node == dest:
                print("BMS:", path)
                return all_paths
            if node not in visited:
                visited.add(node)
                for neighbor in sorted(graph.graph[node], reverse=True):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
            front += 1
        return None

    def DFS(self, graph, start, dest):
        visited = set()
        stack = [(start, [start])]
        all_paths = []
        while stack:
            node, path = stack.pop()
            all_paths.append(path)
            if node == dest:
                print("DFS:", path)
                return all_paths
            if node not in visited:
                visited.add(node)
                for neighbor in sorted(graph.graph[node], reverse=True):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
        return None

    def BFS(self, graph, start, dest):
        visited = set()
        queue = [(start, [start])]
        all_paths = []
        while queue:
            node, path = queue.pop(0)
            all_paths.append(path)
            if node == dest:
                print("BFS: ", path)
                return all_paths
            if node not in visited:
                visited.add(node)
                for neighbor in graph.graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        return None

    def HC(self, graph, start, dest):
        path = []
        total_path = []
        visited = set()
        node = start
        while node != dest:
            path.append(node)
            visited.add(node)
            neighbors = graph.graph[node]
            neighbor_heuristics = [graph.heuristic[neighbor] for neighbor in neighbors]
            best_neighbor = neighbors[neighbor_heuristics.index(min(neighbor_heuristics))]
            if best_neighbor in visited:
                return total_path
            node = best_neighbor
            total_path.append(list(path[:]))
        path.append(dest)
        total_path.append(list(path[:]))
        print("HC: ", path)
        return total_path

    def BS(self, graph, start, dest, bw=1):
        beam = [(graph.heuristic[start], (start, [start]))]
        all_paths = []
        while beam:
            beam.sort(key=lambda x: x[0])
            best_paths = beam[:bw]
            beam = []
            for misc, (node, path) in best_paths:
                all_paths.append(path)
                if node == dest:
                    print("Beam Search: ", path)
                    return all_paths
                for neighbor in graph.graph[node]:
                    if neighbor not in path:
                        heuristic_score = graph.heuristic[neighbor]
                        new_path = path + [neighbor]
                        beam.append((heuristic_score, (neighbor, new_path)))
        return None

    def Oracle(self, graph, start, dest):
        all_paths = []
        all_paths = []
        stack = [(start, [], 0)]  #(node, path, cost)
        while stack:
            current, path, cost = stack.pop()
            all_paths.append(path+[current])
            if current == dest:
                all_paths.append((path + [current], cost))
            else:
                for neighbor, weight in zip(graph.graph[current], graph.weight[current]):
                    if neighbor not in path:
                        stack.append((neighbor, path + [current], cost + weight))
        print("Oracle:", all_paths)
        return all_paths

    def BB(self, graph, start, dest):
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, start, [])]
        all_paths = []

        while priority_queue:
            # Finds the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            all_paths.append(path+[current])
            if current == dest:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(graph.graph[current], graph.weight[current]):
                    if neighbor not in path:
                        if cost+weight<=best_cost:
                            # Add the neighbor to the priority queue with updated cost
                            priority_queue.append((cost + weight, neighbor, path + [current]))

        print("Branch & Bound: ", best_path, best_cost)
        return all_paths

    def EL(self, graph, start, dest):
        best_path = None
        best_cost = float('inf')

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, start, [])]
        all_paths = []

        extended_list = {node: False for node in graph.graph}

        while priority_queue:
            # Finds path with lowest cost in priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)

            all_paths.append(path+[current])

            if current == dest:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(graph.graph[current], graph.weight[current]):
                    if not extended_list[current] and not extended_list[neighbor]:
                        if cost+weight<=best_cost:
                        #Adds neighbor to priority queue with updated cost
                            priority_queue.append((cost + weight, neighbor, path + [current]))
            extended_list[current] = True
        print("Brand & Bound with EL: ", best_path, best_cost)
        return all_paths

    def EH(self, graph, start, dest):
        best_path = None
        best_cost = float('inf')

        priority_queue = [(0, start, [])]
        all_paths = []

        while priority_queue:
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + graph.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + graph.heuristic[priority_queue[min_index][1]]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)

            all_paths.append(path+[current])
            if current == dest:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(graph.graph[current], graph.weight[current]):
                    if neighbor not in path:
                        if cost+weight+graph.heuristic[current]<=best_cost:
                            priority_queue.append((cost + weight, neighbor, path + [current]))

        print("Brand & Bound with EH: ",best_path, best_cost)
        return all_paths

    def Astar(self, graph, start, dest):
        best_path = None
        best_cost = float('inf')

        priority_queue = [(0, start, [])]
        all_paths = []

        extended_list = {node: False for node in graph.graph}

        while priority_queue:
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + graph.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + graph.heuristic[priority_queue[min_index][1]]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            visited = set(path)
            all_paths.append(path+[current])

            if current == dest:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(graph.graph[current], graph.weight[current]):
                    if not extended_list[current] and not extended_list[neighbor] and neighbor not in visited:
                        if cost+weight+graph.heuristic[current]<=best_cost:
                            priority_queue.append((cost + weight, neighbor, path + [current]))
            extended_list[current] = True

        print("A-Star:", best_path, best_cost)
        return all_paths

    # def AOStar(self, graph, start, dest):
    #     open_list = [(start, 0)]  #Priority queue to store nodes and their associated costs
    #     closed_list = set()  #Set to keep track of explored nodes
    #     g_values = {node: float('inf') for node in graph.graph}  #Cost from the start node to each node
    #     g_values[start] = 0
    #     parent = {}  #Stores parent of each node
    #     all_paths = []

    #     while open_list:
    #         current_node, current_cost = open_list.pop(0)

    #         if current_node == dest:
    #             path = [current_node]
    #             all_paths.append(path+[current_node])
    #             while current_node in parent:
    #                 current_node = parent[current_node]
    #                 path.insert(0, current_node)
    #             print(path)

    #         if current_node in closed_list:
    #             continue

    #         closed_list.add(current_node)

    #         for i, neighbor in enumerate(graph.graph[current_node]):
    #             cost_to_neighbor = current_cost + graph.weight[current_node][i]

    #             if cost_to_neighbor < g_values[neighbor]:
    #                 g_values[neighbor] = cost_to_neighbor
    #                 f_value = cost_to_neighbor + graph.heuristic[neighbor]
    #                 open_list.append((neighbor, f_value))
    #                 open_list.sort(key=lambda x: x[1])
    #                 parent[neighbor] = current_node

    #     return all_paths

    def AOStar(self, graph, start, dest):
        open_list = [(0, [start])]
        closed_list = set()
        all_paths = []

        while open_list:
            open_list.sort(key=lambda x: x[0])
            current_cost, path = open_list.pop(0)
            current_node = path[-1]

            if current_node == dest:
                all_paths.append(path)
                print("AO-Star:", path)
            else:
                if current_node not in closed_list:
                    closed_list.add(current_node)
                    for neighbor, weight in zip(graph.graph[current_node], graph.weight[current_node]):
                        if neighbor not in path:
                            new_path = path + [neighbor]
                            new_cost = current_cost + weight + graph.heuristic[neighbor]  # AO* priority
                            open_list.append((new_cost, new_path))

        return all_paths

    def BestFirstSearch(self, graph, start, dest):
        best_path = None
        priority_queue = [(graph.heuristic[start], start, [])]#(heuristic, node, path)
        all_paths = []

        while priority_queue:
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            heuristic, current, path = priority_queue.pop(min_index)

            all_paths.append(path+[current])

            if current == dest:
                best_path = path + [current]
                print(best_path)
                return all_paths
            else:
                for neighbor in graph.graph[current]:
                    if neighbor not in path:
                        priority_queue.append((graph.heuristic[neighbor], neighbor, path + [current]))

        print("Best First Search:", best_path)
        return all_paths

class Visualization:

    def plot_viz(self, graph, start, dest, traversal_algorithm, bw = 2):
        G = nx.Graph()
        for node, neighbors in graph.graph.items():
            for neighbor, weight in zip(neighbors, graph.weight[node]):
                G.add_edge(node, neighbor, weight=weight)

        if traversal_algorithm.__name__ == "BS":
            paths = traversal_algorithm(graph, start, dest, bw)
        else:
            paths = traversal_algorithm(graph, start, dest)
        pos = nx.planar_layout(G)

        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            node_labels = {node: f"{node}\nH_val:{graph.heuristic[node]}" for node in G.nodes()}
            nx.draw(G, pos, with_labels=True, node_size=1000, font_size=10, node_color='yellow', font_color='black', font_weight='bold',labels = node_labels, ax=ax)
            edge_labels = {(node, neighbor): G[node][neighbor]['weight'] for node, neighbor in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=10, ax=ax)

            if frame < len(paths):
                path = paths[frame]
                path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)

        ani = FuncAnimation(fig, update, frames=len(paths), repeat=False, interval=500)
        plt.show()

graph = Graph()
algo = Algorithm()

graph.addEdge('P','R',4)
graph.addEdge('P','C',4)
graph.addEdge('P','A',4)

graph.addEdge('R','C',2)
graph.addEdge('R','E',5)

graph.addEdge('E','S',1)
graph.addEdge('E','U',5)

graph.addEdge('N','S',6)
graph.addEdge('N','L',5)

graph.addEdge('L','M',2)

graph.addEdge('M','U',5)
graph.addEdge('M','C',6)
graph.addEdge('M','A',3)

graph.addEdge('U','S',4)
graph.addEdge('U','C',3)


graph.add_heuristics('P',10)
graph.add_heuristics('R',8)
graph.add_heuristics('E',3)
graph.add_heuristics('S',0)
graph.add_heuristics('N',6)
graph.add_heuristics('L',9)
graph.add_heuristics('M',9)
graph.add_heuristics('A',11)
graph.add_heuristics('U',4)
graph.add_heuristics('C',6)

# graph.addEdge('S','A',3)
# graph.addEdge('S','B',5)

# graph.addEdge('A','D',3)
# graph.addEdge('A','B',4)

# graph.addEdge('B','C',4)

# graph.addEdge('C','E',6)
# graph.addEdge('D','G',5)

# graph.add_heuristics('A',7)
# graph.add_heuristics('B',6)
# graph.add_heuristics('C',7)

# Visualization().plot_viz(graph, 'P', 'S', algo.BMS)
# Visualization().plot_viz(graph, 'P', 'S', algo.BFS)
# Visualization().plot_viz(graph, 'P', 'S', algo.DFS)
# Visualization().plot_viz(graph, 'P', 'S', algo.BS)
Visualization().plot_viz(graph, 'P', 'S', algo.HC)
# Visualization().plot_viz(graph, 'P', 'S', algo.BB)
# Visualization().plot_viz(graph, 'P', 'S', algo.EL)
# Visualization().plot_viz(graph, 'P', 'S', algo.EH)
# Visualization().plot_viz(graph, 'P', 'S', algo.Oracle)
# Visualization().plot_viz(graph, 'P', 'S', algo.Astar)
# Visualization().plot_viz(graph, 'P', 'S', algo.AOStar)
# Visualization().plot_viz(graph, 'P', 'S', algo.BestFirstSearch)