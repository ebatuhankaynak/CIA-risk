import networkx as nx

from graph_creator import MockGraphCreator

graph_creator = MockGraphCreator()

G = graph_creator.create_graph(vertex_count=20, edge_count=30)
pagerank = nx.pagerank(G)
pagerank_scores = {k: round(v, 4) for k, v in pagerank.items()}

G = graph_creator.scramble_graph_edges(G)
pagerank = nx.pagerank(G)
new_pagerank_scores = {k: round(v, 4) for k, v in pagerank.items()}

pagerank_changes = {node: new_pagerank_scores[node] - pagerank_scores[node] for node in new_pagerank_scores}

print(pagerank_changes)