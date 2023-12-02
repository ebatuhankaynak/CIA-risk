import networkx as nx
from utils import Utils
from tqdm import tqdm

from graph_creator import MockGraphCreator

graph_creator = MockGraphCreator()

filename = "raw/Cli.json"
commit_data = Utils.read_json(filename)
#changed_files = Utils.get_commit_info(commit_data, sha)

commit_ids = list(commit_data.keys())
pg_scores = {}
for cid in tqdm(commit_ids, "Calculating Pagerank for commits"):
    commit_files = Utils.get_project_files(commit_data, cid)

    if len(commit_files) > 1:
        G = graph_creator.create_graph(commit_files, edge_count=20)
        pagerank = nx.pagerank(G)

        pagerank_scores = {k: round(v, 4) for k, v in pagerank.items()}
        pg_scores[cid] = pagerank_scores
    else:
        pg_scores[cid] = {}

pg_diffs = {}
for i in tqdm(range(len(commit_ids) - 1), "Calculating Pagerank differences between commits"):
    current_cid = commit_ids[i]
    next_cid = commit_ids[i + 1]

    current_scores = pg_scores[current_cid]
    next_scores = pg_scores[next_cid]
    score_differences = {node: next_scores.get(node, 0) - current_scores.get(node, 0) for node in current_scores}

    pg_diffs[(current_cid, next_cid)] = score_differences
