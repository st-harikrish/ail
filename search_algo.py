class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node, connections):
        self.graph[node] = connections

def print_graph(graph):
    for node, connections in graph.graph.items():
        print(f"{node} -> {', '.join(connections)}")

def take_input():
    graph = Graph()

    nodes = input("Enter nodes separated by spaces: ").split()
    for node in nodes:
        connections = input(f"Enter connections for node {node} separated by spaces (or 'None'): ").split()
        connections = [c for c in connections if c != 'None']  # Filter out 'None' connections
        graph.add_edge(node, connections)

    cost = {}
    heuristics = {}

    for node in nodes:
        cost[node] = float(input(f"Enter cost for node {node}: "))
        heuristics[node] = float(input(f"Enter heuristic for node {node}: "))

    return graph, cost, heuristics

def british_museum(graph, start, goal):
    visited = set()
    queue = [start]

    while queue:
        node = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {node}")
            return

        if node not in visited:
            visited.add(node)
            queue.extend(graph.graph[node])

    print("Goal not found")

def breadth_first_search(graph, start, goal):
    visited = set()
    queue = [(start, [start])]

    while queue:
        (node, path) = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                queue.append((neighbor, path + [neighbor]))

    print("Goal not found")

def depth_first_search(graph, start, goal):
    visited = set()

    def dfs_helper(node, path):
        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return True

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if dfs_helper(neighbor, path + [neighbor]):
                    return True

        return False

    if not dfs_helper(start, [start]):
        print("Goal not found")


def hill_climbers(graph, start, goal, heuristic):
    current_node = start

    while current_node != goal:
        neighbors = graph.graph[current_node]
        next_node = None

        for neighbor in neighbors:
            if neighbor in heuristic and heuristic[neighbor] < heuristic[current_node]:
                if next_node is None or heuristic[neighbor] < heuristic[next_node]:
                    next_node = neighbor

        if next_node is None:
            print("Stuck in local minima, cannot reach the goal.")
            return

        print(f"Current Node: {current_node}, Next Node: {next_node}")
        current_node = next_node

        if current_node == goal:
            print(f"Goal reached: {current_node}")
            return

    print("Goal not found")

def beam_search(graph, start, goal, k, heuristic):
    visited = set()
    queue = [(start, [start], 0)]

    while queue:
        queue.sort(key=lambda x: x[2])
        queue = queue[:k]  # Keep only top k nodes

        (node, path, cost) = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if neighbor not in path:
                    new_cost = cost + heuristic.get(neighbor, 0)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_cost))

    print("Goal not found")


def oracle(graph, start, goal):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()

        if node == goal:
            print(f"Goal reached: {node}")
            return

        if node not in visited:
            visited.add(node)
            stack.extend(graph.graph[node])

    print("Goal not found")

def oracle_with_cost(graph, start, goal, cost):
    visited = set()
    stack = [(start, [start], 0)]

    while stack:
        (node, path, path_cost) = stack.pop()

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if neighbor not in path:
                    new_cost = path_cost + cost[neighbor]
                    new_path = path + [neighbor]
                    stack.append((neighbor, new_path, new_cost))

    print("Goal not found")

def branch_and_bound(graph, start, goal, cost):
    visited = set()
    queue = [(start, [start], 0)]

    while queue:
        queue.sort(key=lambda x: x[2])
        (node, path, path_cost) = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if neighbor not in path:
                    new_cost = path_cost + cost[neighbor]  # Use cost as a dictionary
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_cost))

    print("Goal not found")


def branch_and_bound_extended(graph, start, goal, cost):
    visited = set()
    queue = [(start, [start], 0)]

    while queue:
        queue.sort(key=lambda x: x[2])
        (node, path, path_cost) = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if neighbor not in path:
                    new_cost = path_cost + cost[neighbor]  # Use cost as a dictionary
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_cost))

    print("Goal not found")


def branch_and_bound_heuristics(graph, start, goal, cost, heuristics):
    visited = set()
    queue = [(start, [start], 0, 0)]

    while queue:
        queue.sort(key=lambda x: x[2] + x[3])
        (node, path, path_cost, heuristic_cost) = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if neighbor not in path:
                    new_cost = path_cost + cost[neighbor]
                    new_heuristic = heuristics[neighbor]
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_cost, new_heuristic))

    print("Goal not found")


def a_star(graph, start, goal, cost):
    visited = set()
    queue = [(start, [start], 0)]

    while queue:
        queue.sort(key=lambda x: x[2] + cost[x[0]])  # Use cost as a dictionary
        (node, path, path_cost) = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if neighbor not in path:
                    new_cost = path_cost + cost[neighbor]  # Use cost as a dictionary
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_cost))

    print("Goal not found")


    print("Goal not found")
def ao_star(graph, start, goal, cost, heuristics):
    visited = set()
    queue = [(start, [start], 0, heuristics[start])]

    while queue:
        queue.sort(key=lambda x: x[3])  # Sort by heuristic value
        (node, path, path_cost, heuristic_cost) = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if neighbor not in path:
                    new_cost = path_cost + cost[neighbor]
                    new_heuristic = heuristics[neighbor]
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_cost, new_heuristic))

    print("Goal not found")

def best_first_search(graph, start, goal, heuristics):
    visited = set()
    queue = [(start, [start], heuristics[start])]

    while queue:
        queue.sort(key=lambda x: x[2])  # Sort by heuristic value
        (node, path, heuristic_cost) = queue.pop(0)

        if node == goal:
            print(f"Goal reached: {' -> '.join(path)}")
            return

        if node not in visited:
            visited.add(node)
            for neighbor in graph.graph[node]:
                if neighbor not in path:
                    new_heuristic = heuristics[neighbor]
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, new_heuristic))

    print("Goal not found")


# Main program
graph, cost, heuristics = take_input()

start_node = input("Enter the start node: ")
goal_node = input("Enter the goal node: ")

print_graph(graph)

print("British Museum")
british_museum(graph, start_node, goal_node)
print("Breadth First Search")
breadth_first_search(graph, start_node, goal_node)
print("Depth First Search")
depth_first_search(graph, start_node, goal_node)
print("Hill Climbers")
hill_climbers(graph, start_node, goal_node, heuristics)
print("Beam Search")
beam_search(graph, start_node, goal_node, 2, heuristics)  # Replace k with the beam width
print("Oracle")
oracle(graph, start_node, goal_node)
print("Oracle with Cost")
oracle_with_cost(graph, start_node, goal_node, cost)
print("Branch and Bound")
branch_and_bound(graph, start_node, goal_node,cost)
print("Branch and bound with Extended List")
branch_and_bound_extended(graph, start_node, goal_node,cost)
print("Branch and bound with Heuristics")
branch_and_bound_heuristics(graph, start_node, goal_node,cost,heuristics)
print("A star")
a_star(graph, start_node, goal_node,cost)
print("AO* Best First Search")
ao_star(graph, start_node, goal_node, cost, heuristics)
print("Best First Search")
best_first_search(graph, start_node, goal_node, heuristics)