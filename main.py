import pandas as pd
import json

from data_handler import DataHandler

dh = DataHandler()

requested_features = [
    "global_graph_scores", 
    "changed_graph_scores", 
    "global_graph_diffs", 
    "changed_graph_diffs",
]

features_per_cid = dh.create_features("Cli", requested_features)
commit_ids, labels = dh.create_labels("Cli")

def flatten_features(features_dict):
    flattened_list = []
    for key in features_dict:
        flattened_list.extend(features_dict[key])
    return flattened_list

for idx, k in enumerate(features_per_cid.keys()):
    print(k)
    print(commit_ids[idx])

"""print(commit_ids)
print(idx, len(commit_ids))"""