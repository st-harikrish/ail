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
    def AOStar(self, graph, start, dest):
        open_list = [(start, 0)]  #Priority queue to store nodes and their associated costs
        closed_list = set()  #Set to keep track of explored nodes
        g_values = {node: float('inf') for node in graph.graph}  #Cost from the start node to each node
        g_values[start] = 0
        parent = {}  #Stores parent of each node

        while open_list:
            current_node, current_cost = open_list.pop(0)

            if current_node == dest:
                path = [current_node]
                while current_node in parent:
                    current_node = parent[current_node]
                    path.insert(0, current_node)
                return path

            if current_node in closed_list:
                continue

            closed_list.add(current_node)

            for i, neighbor in enumerate(graph.graph[current_node]):
                cost_to_neighbor = current_cost + graph.weight[current_node][i]

                if cost_to_neighbor < g_values[neighbor]:
                    g_values[neighbor] = cost_to_neighbor
                    f_value = cost_to_neighbor + graph.heuristic[neighbor]
                    open_list.append((neighbor, f_value))
                    open_list.sort(key=lambda x: x[1])
                    parent[neighbor] = current_node

        return

graph = Graph()
algo = Algorithm()

graph.addEdge('S','A',3)
graph.addEdge('S','B',5)

graph.addEdge('A','D',3)
graph.addEdge('A','B',4)

graph.addEdge('B','C',4)

graph.addEdge('C','E',6)
graph.addEdge('D','G',5)

graph.add_heuristics('A',7)
graph.add_heuristics('B',6)
graph.add_heuristics('C',7)

start_node = 'S'
end_node = 'G'

print(algo.AOStar(graph, start_node, end_node))