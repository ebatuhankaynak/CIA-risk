import random
import networkx as nx

class MockGraphCreator:
    def generate_method_name(self):
        methods = ['getData', 'processData', 'saveData', 'loadData', 'init', 'start', 'stop', 'calculate']
        return random.choice(methods)

    def generate_class_name(self):
        classes = ['Manager', 'Controller', 'Service', 'Repository', 'Helper', 'Worker']
        return random.choice(classes)

    def generate_package_name(self):
        packages = ['com.example.app', 'com.example.lib', 'com.example.utils', 'com.example.core']
        return random.choice(packages)

    def create_graph(self, vertex_count, edge_count, verbose=False):
        nodes = set()
        for _ in range(vertex_count):
            package = self.generate_package_name()
            class_name = self.generate_class_name()
            method = self.generate_method_name()
            node = f"{package}.{class_name}.{method}"
            nodes.add(node)

        edges = []
        for _ in range(edge_count):
            source, target = random.sample(nodes, 2)
            edges.append((source, target))

        call_graph = '\n'.join([f"{src} -> {tgt}" for src, tgt in edges])
        if verbose:
            print(call_graph)
        
        G = nx.DiGraph()
        for src, tgt in edges:
            G.add_edge(src, tgt)

        return G
    
    def create_graph(self, nodes, edge_count, verbose=False):
        edges = []

        for node in nodes: # Pair each node with a random other node
            target = random.choice([n for n in nodes if n != node])
            edges.append((node, target))

        for _ in range(edge_count):
            source, target = random.sample(nodes, 2)
            edges.append((source, target))

        call_graph = '\n'.join([f"{src} -> {tgt}" for src, tgt in edges])
        if verbose:
            print(call_graph)
        
        G = nx.DiGraph()
        for src, tgt in edges:
            G.add_edge(src, tgt)

        return G
    
    def scramble_graph_edges(self, graph):
        nodes = list(graph.nodes())
        scrambled_edges = []

        while len(scrambled_edges) < graph.number_of_edges():
            src, tgt = random.sample(nodes, 2)
            new_edge = (src, tgt)

            if new_edge not in scrambled_edges:
                scrambled_edges.append(new_edge)

        scrambled_graph = nx.DiGraph()
        scrambled_graph.add_edges_from(scrambled_edges)
        return scrambled_graph
    
class GraphCreator:
    def create_graph(self):
        raise NotImplementedError
