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
y = dh.create_labels("Cli")

def flatten_features(features_dict):
    flattened_lists = {}
    for cid in features_dict:
        flattened_lists[cid] = []
        for feat_key in features_dict[cid]:
            flattened_lists[cid].extend(features_dict[cid][feat_key])
    return flattened_lists

x = flatten_features(features_per_cid)

print(x)
print(y)
