import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import re
import csv
import os

from graph_creator import MockGraphCreator, GraphCreator
from utils import MockUtils

class MockDataHandler:
    def calculate_graph_metrics(self, G):
        pagerank = nx.pagerank(G)
        centrality = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        clustering_coefficient = nx.clustering(G)

        return pagerank, centrality, closeness, clustering_coefficient

    def calculate_aggregate_metrics(self, metrics):
        if not metrics:
            return {'mean': 0, 'std': 0, 'median': 0}

        values = np.array(list(metrics.values()))

        return {
            'mean': np.nanmean(values),
            'std': np.nanstd(values),
            'median': np.nanmedian(values),
        }
    
    def create_features(self, project_name, requested_features):
        filename = f"raw/{project_name}.json"
        commit_data = MockUtils.read_json(filename)

        graph_creator = MockGraphCreator()

        commit_ids = list(commit_data.keys())

        global_graph_scores = {}
        changed_graph_scores = {}
        bad_commits = []
        for cid in tqdm(commit_ids, "Calculating Metrics for commits"):
            commit_files = MockUtils.get_project_files(commit_data, cid)
            changed_files = MockUtils.get_commit_info(commit_data, cid)

            if len(commit_files) > 1:
                G = graph_creator.create_graph(commit_files, edge_count=0)

                pagerank, centrality, closeness, clustering_coeff = self.calculate_graph_metrics(G)

                global_graph_scores[cid] = {
                    'pagerank': self.calculate_aggregate_metrics(pagerank),
                    'centrality': self.calculate_aggregate_metrics(centrality),
                    'closeness': self.calculate_aggregate_metrics(closeness),
                    'clustering_coefficient': self.calculate_aggregate_metrics(clustering_coeff)
                }

                changed_graph_scores[cid] = {
                    'pagerank': self.calculate_aggregate_metrics({file: pagerank[file] for file in changed_files if file in pagerank}),
                    'centrality': self.calculate_aggregate_metrics({file: centrality[file] for file in changed_files if file in centrality}),
                    'closeness': self.calculate_aggregate_metrics({file: closeness[file] for file in changed_files if file in closeness}),
                    'clustering_coefficient': self.calculate_aggregate_metrics({file: clustering_coeff[file] for file in changed_files if file in clustering_coeff}),
                }
            else:
                bad_commits.append(cid)

        global_graph_diffs = {}
        changed_graph_diffs = {}
        for i in tqdm(range(len(commit_ids) - 1), "Calculating Differences between commits"):
            current_cid = commit_ids[i]
            next_cid = commit_ids[i + 1]

            if current_cid in bad_commits or next_cid in bad_commits:
                continue

            def calc_diff(current_scores, next_scores):
                score_differences = {}
                for metric in current_scores:
                    score_differences[metric] = {key: next_scores[metric].get(key, 0) - current_scores[metric].get(key, 0) for key in current_scores[metric]}

                return score_differences

            global_graph_diffs[next_cid] = calc_diff(global_graph_scores[current_cid], global_graph_scores[next_cid])
            changed_graph_diffs[next_cid] = calc_diff(changed_graph_scores[current_cid], changed_graph_scores[next_cid])

        metrics_order = ['pagerank', 'centrality', 'closeness', 'clustering_coefficient']
        def linearize_metrics(metric_dict):
            linear_vectors = {}
            for cid, metrics in metric_dict.items():
                linear_vector = []
                for metric in metrics_order:
                    linear_vector.extend([metrics[metric][key] for key in sorted(metrics[metric].keys())])

                linear_vectors[cid] = linear_vector
            return linear_vectors
        
        def attach_cid_to_features(metrics_dicts):
            linearized_metrics = {name: linearize_metrics(metric_dict) for name, metric_dict in metrics_dicts.items()}

            consolidated_features = {}
            for cid in commit_ids:
                consolidated_features[cid] = {name: linearized_metrics[name].get(cid, []) for name in linearized_metrics}

            return consolidated_features

        metrics_dicts = {
            "global_graph_scores": global_graph_scores,
            "changed_graph_scores": changed_graph_scores,
            "global_graph_diffs": global_graph_diffs,
            "changed_graph_diffs": changed_graph_diffs
        }

        filtered_metrics_dicts = {name: metrics for name, metrics in metrics_dicts.items() if name in requested_features}
        return attach_cid_to_features(filtered_metrics_dicts)
    
    
    def create_labels(self, project_name):
        project_name_to_dbname = {
            "Cli": "org.apache:commons-cli"
        }

        bic_dataset_path = f"fault_induce.txt"
        bic_dataset = pd.read_csv(bic_dataset_path, delimiter='|')

        directory_path = f"caller_callee_outputs/{project_name}/"
        commit_ids = [file_name.split('.')[0] for file_name in os.listdir(directory_path)]
        print(bic_dataset)
        bic_dataset_for_pid = bic_dataset[bic_dataset['PROJECT_ID'] == project_name_to_dbname[project_name]]

        bic_commit_ids = set(bic_dataset_for_pid['FAULT_INDUCING_COMMIT_HASH'])
        print(len(bic_commit_ids), len(commit_ids))
        
        # Assuming bic_commit_ids and commit_ids are both sets
        common_commit_ids = bic_commit_ids.intersection(commit_ids)
        labels = [1 if commit_id in common_commit_ids else 0 for commit_id in commit_ids]

        return labels
    
    def read_and_parse_csv(self, file_path):
        regex = r'(org\.apache\.[\w\.]+\([^)]*\)),(org\.apache\.[\w\.]+\([^)]*\))'
        data = []
        with open(file_path, 'r') as file:
            next(file)
            for line in file:
                match = re.match(regex, line.strip())
                if match:
                    caller, callee = match.groups()
                    data.append({'Caller': caller, 'Callee': callee})
        return pd.DataFrame(data)

    def create_example_project(self):
        file_path = 'example_cc.csv'
        project_name = "Cli"

        df = self.read_and_parse_csv(file_path)
        filename = f"raw/{project_name}.json"
        commit_data = MockUtils.read_json(filename)
        commit_ids = list(commit_data.keys())

        for i, commit_id in enumerate(commit_ids, start=1):
            new_df = df.iloc[:i]
            with open(f"caller_callee_outputs/example/{commit_id}.csv", 'w', newline='', encoding='utf-8') as f:
                f.write("Caller,Callee\n")
                for index, row in new_df.iterrows():
                    f.write(f"{row['Caller']},{row['Callee']}\n")
            

    def create_features_from_cc(self, project_name, requested_features):
        filename = f"raw/{project_name}.json"
        commit_data = MockUtils.read_json(filename)

        graph_creator = GraphCreator()

        commit_ids = list(commit_data.keys())

        global_graph_scores = {}
        changed_graph_scores = {}
        global_graph_diffs = {}
        changed_graph_diffs = {}

        def extract_unique_methods(df1, df2):
            diff_df = pd.concat([df1, df2]).drop_duplicates(keep=False)
            unique_methods = pd.unique(diff_df[['Caller', 'Callee']].values.ravel('K'))
            return list(unique_methods)
        
        for i in tqdm(range(len(commit_ids) - 1), "Calculating Global Metrics for commits"):
            current_cid = commit_ids[i]
            next_cid = commit_ids[i + 1]

            #print(current_cid)
            try:
                current_cc_pair_df = self.read_and_parse_csv(f'caller_callee_outputs/{project_name}/{current_cid}.csv')
                next_cc_pair_df = self.read_and_parse_csv(f'caller_callee_outputs/{project_name}/{next_cid}.csv')
            except FileNotFoundError:
                continue
            changed_files = extract_unique_methods(current_cc_pair_df, next_cc_pair_df)

            G = graph_creator.create_graph(current_cc_pair_df)

            pagerank, centrality, closeness, clustering_coeff = self.calculate_graph_metrics(G)

            global_graph_scores[current_cid] = {
                'pagerank': self.calculate_aggregate_metrics(pagerank),
                'centrality': self.calculate_aggregate_metrics(centrality),
                'closeness': self.calculate_aggregate_metrics(closeness),
                'clustering_coefficient': self.calculate_aggregate_metrics(clustering_coeff)
            }

            """print(global_graph_scores[current_cid])
            input()"""

            changed_graph_scores[current_cid] = {
                'pagerank': self.calculate_aggregate_metrics({file: pagerank[file] for file in changed_files if file in pagerank}),
                'centrality': self.calculate_aggregate_metrics({file: centrality[file] for file in changed_files if file in centrality}),
                'closeness': self.calculate_aggregate_metrics({file: closeness[file] for file in changed_files if file in closeness}),
                'clustering_coefficient': self.calculate_aggregate_metrics({file: clustering_coeff[file] for file in changed_files if file in clustering_coeff}),
            }

        for i in tqdm(range(len(commit_ids) - 2), "Calculating Change Metrics for commits"):
            current_cid = commit_ids[i]
            next_cid = commit_ids[i + 1]

            def calc_diff(current_scores, next_scores):
                score_differences = {}
                for metric in current_scores:
                    score_differences[metric] = {key: next_scores[metric].get(key, 0) - current_scores[metric].get(key, 0) for key in current_scores[metric]}

                return score_differences

            try:
                global_graph_diffs[next_cid] = calc_diff(global_graph_scores[current_cid], global_graph_scores[next_cid])
                changed_graph_diffs[next_cid] = calc_diff(changed_graph_scores[current_cid], changed_graph_scores[next_cid])
            except KeyError:
                continue
  
        metrics_order = ['pagerank', 'centrality', 'closeness', 'clustering_coefficient']
        def linearize_metrics(metric_dict):
            linear_vectors = {}
            for cid, metrics in metric_dict.items():
                linear_vector = []
                for metric in metrics_order:
                    linear_vector.extend([metrics[metric][key] for key in sorted(metrics[metric].keys())])

                linear_vectors[cid] = linear_vector
            return linear_vectors
        
        def attach_cid_to_features(metrics_dicts):
            linearized_metrics = {name: linearize_metrics(metric_dict) for name, metric_dict in metrics_dicts.items()}

            consolidated_features = {}
            for cid in commit_ids:
                consolidated_features[cid] = {name: linearized_metrics[name].get(cid, []) for name in linearized_metrics}

            return consolidated_features

        metrics_dicts = {
            "global_graph_scores": global_graph_scores,
            "changed_graph_scores": changed_graph_scores,
            "global_graph_diffs": global_graph_diffs,
            "changed_graph_diffs": changed_graph_diffs
        }

        filtered_metrics_dicts = {name: metrics for name, metrics in metrics_dicts.items() if name in requested_features}
        return attach_cid_to_features(filtered_metrics_dicts)
