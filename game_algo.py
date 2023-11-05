import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

def visualize_game_tree(root_node):
    G = nx.DiGraph()
    stack = [(None, root_node)]

    while stack:
        parent, node = stack.pop()
        G.add_node(node)
        if parent:
            G.add_edge(parent, node)

        for child in node.children:
            stack.append((node, child))

    pos = nx.spectral_layout(G)
    labels = {node: str(node.value) for node in G.nodes}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, node_color='yellow', font_size=10)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{u.value}->{v.value}' for u, v in G.edges})
    plt.show()

#Alpha-Beta Pruning
def alpha_beta(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or not node.children:
        return node.value

    if maximizing_player:
        max_eval = -float('inf')
        for child in node.children:
            max_eval = max(max_eval, alpha_beta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, max_eval)
            if alpha >= beta:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.children:
            min_eval = min(min_eval, alpha_beta(child, depth - 1, alpha, beta, True))
            beta = min(beta, min_eval)
            if alpha >= beta:
                break
        return min_eval

#Min-Max
def minimax(node, depth, maximizing_player):
    if depth == 0 or not node.children:
        return node.value

    if maximizing_player:
        max_eval = -float('inf')
        for child in node.children:
            eval = minimax(child, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.children:
            eval = minimax(child, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval


#Game Tree
# root = Node(0)
# child1 = Node(1)
# child2 = Node(2)
# child3 = Node(5)
# child4 = Node(6)
# child5 = Node(7)
# child6 = Node(8)

# root.add_child(child1)
# root.add_child(child2)
# child1.add_child(child3)
# child1.add_child(child4)
# child2.add_child(child5)
# child2.add_child(child6)

root = Node(0)
child1 = Node(1)
child2 = Node(2)
child3 = Node(5)
child4 = Node(6)
child5 = Node(7)

root.add_child(child1)
root.add_child(child2)
child1.add_child(child3)
child1.add_child(child4)
child4.add_child(child5)  # Creating an unbalanced branch

minimax_value = minimax(root, 3, True)
print("Minimax Value: ", minimax_value)

visualize_game_tree(root)

alpha_beta_value = alpha_beta(root, 3, -float('inf'), float('inf'), True)
print("Alpha-Beta Value: ", alpha_beta_value)

visualize_game_tree(root)