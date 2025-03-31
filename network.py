import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class GraphVisualizer:
    def __init__(self, num_nodes=5, num_edges=6):
        """
        Inicializuje graf s náhodnými hranami.
        :param num_nodes: Počet uzlů v grafu.
        :param num_edges: Počet hran v grafu.
        """
        self.graph = nx.Graph()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self._generate_graph()

    def _generate_graph(self):
        """Vygeneruje náhodný graf s daným počtem uzlů a hran."""
        self.graph.add_nodes_from(range(self.num_nodes))
        while len(self.graph.edges) < self.num_edges:
            u, v = np.random.choice(self.num_nodes, 2, replace=False)
            self.graph.add_edge(u, v)

    def show(self):
        """Zobrazí graf pomocí Matplotlibu."""
        plt.figure(figsize=(6, 6))
        nx.draw(self.graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=12)
        plt.show()


# Příklad použití
gv = GraphVisualizer(num_nodes=20, num_edges=45)
gv.show()