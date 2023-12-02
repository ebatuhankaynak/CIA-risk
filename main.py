import networkx as nx
from utils import Utils

from graph_creator import GraphCreator

graph_creator = GraphCreator()

commit_ids = ["abc", "def"]
pg_scores = {}
for cid in commit_ids:
    G = graph_creator.create_graph()
    pagerank = nx.pagerank(G)

    pagerank_scores = {k: round(v, 4) for k, v in pagerank.items()}
    pg_scores[cid] = pagerank_scores


pg_diffs = {}
for i in range(len(commit_ids) - 1):
    current_cid = commit_ids[i]
    next_cid = commit_ids[i + 1]

    current_scores = pg_scores[current_cid]
    next_scores = pg_scores[next_cid]
    score_differences = {node: next_scores.get(node, 0) - current_scores.get(node, 0) for node in current_scores}

    pg_diffs[(current_cid, next_cid)] = score_differences

print("PageRank Score Differences between Commits:", pg_diffs)
