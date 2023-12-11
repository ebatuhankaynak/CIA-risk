from data_handler import MockDataHandler

dh = MockDataHandler()

requested_features = [
    "global_graph_scores", 
    "changed_graph_scores", 
    "global_graph_diffs", 
    "changed_graph_diffs",
]

features_per_cid = dh.create_features("raw/Cli.json", requested_features)

def flatten_features(features_dict):
    flattened_list = []
    for key in features_dict:
        flattened_list.extend(features_dict[key])
    return flattened_list

for k,v in features_per_cid.items():
    print(k, flatten_features(v))