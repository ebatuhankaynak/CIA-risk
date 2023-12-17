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
            "changed_graph_diffs": changed_graph_diffs,
        }

        filtered_metrics_dicts = {name: metrics for name, metrics in metrics_dicts.items() if name in requested_features}
        return attach_cid_to_features(filtered_metrics_dicts)
    
    
    def create_labels(self, project_name):
        project_name_to_dbname = {
            "Cli": "org.apache:commons-cli",
            "Fileupload": "org.apache:commons-fileupload",
            "Beanutils": "org.apache:beanutils",
            "Codec": "org.apache:codec"
        }

        bic_dataset_path = f"fault_induce.txt"
        bic_dataset = pd.read_csv(bic_dataset_path, delimiter='|')

        directory_path = f"caller_callee_outputs/{project_name}/"
        commit_ids = [file_name.split('.')[0] for file_name in os.listdir(directory_path)]
        bic_dataset_for_pid = bic_dataset[bic_dataset['PROJECT_ID'] == project_name_to_dbname[project_name]]

        bic_commit_ids = set(bic_dataset_for_pid['FAULT_INDUCING_COMMIT_HASH'])
        print(len(bic_dataset_for_pid), len(bic_commit_ids), len(commit_ids))
        
        common_commit_ids = bic_commit_ids.intersection(commit_ids)
        labels = [1 if commit_id in common_commit_ids else 0 for commit_id in commit_ids]

        return labels
    
    def read_and_parse_csv(self, file_path, project_name):
        if project_name == "Cli":
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
        else:
            df = pd.read_csv(file_path, delimiter="\t")
            return df

    def create_example_project(self):
        file_path = 'example_cc.csv'
        project_name = "Cli"

        df = self.read_and_parse_csv(file_path, project_name)
        filename = f"raw/{project_name}.json"
        commit_data = MockUtils.read_json(filename)
        commit_ids = list(commit_data.keys())

        for i, commit_id in enumerate(commit_ids, start=1):
            new_df = df.iloc[:i]
            with open(f"caller_callee_outputs/example/{commit_id}.csv", 'w', newline='', encoding='utf-8') as f:
                f.write("Caller,Callee\n")
                for index, row in new_df.iterrows():
                    f.write(f"{row['Caller']},{row['Callee']}\n")
            

    def create_features_from_cc(self, project_name):
        filename = f"raw/{project_name}.json"
        commit_data = MockUtils.read_json(filename)

        graph_creator = GraphCreator()

        commit_ids = list(commit_data.keys())

        global_graph_scores = {}
        changed_graph_scores = {}
        global_graph_diffs = {}
        changed_graph_diffs = {}
        commit_summary = {}

        with open(filename, 'r') as file:
            commit_json = json.load(file)

        def extract_unique_methods(df1, df2):
            diff_df = pd.concat([df1, df2]).drop_duplicates(keep=False)
            unique_methods = pd.unique(diff_df[['Caller', 'Callee']].values.ravel('K'))
            return list(unique_methods)
        
        vertex_change = []
        edge_change = []
        prev_G = None
        for i in tqdm(range(len(commit_ids) - 1), "Calculating Global Metrics for commits"):
            current_cid = commit_ids[i]
            next_cid = commit_ids[i + 1]

            try:
                current_cc_pair_df = self.read_and_parse_csv(f'caller_callee_outputs/{project_name}/{current_cid}.csv', project_name)
                next_cc_pair_df = self.read_and_parse_csv(f'caller_callee_outputs/{project_name}/{next_cid}.csv', project_name)
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

            changed_graph_scores[current_cid] = {
                'pagerank': self.calculate_aggregate_metrics({file: pagerank[file] for file in changed_files if file in pagerank}),
                'centrality': self.calculate_aggregate_metrics({file: centrality[file] for file in changed_files if file in centrality}),
                'closeness': self.calculate_aggregate_metrics({file: closeness[file] for file in changed_files if file in closeness}),
                'clustering_coefficient': self.calculate_aggregate_metrics({file: clustering_coeff[file] for file in changed_files if file in clustering_coeff}),
            }

            files_changed = len(commit_json[current_cid].get('files_changed', []))

            additions = 0
            deletions = 0

            for file in commit_json[current_cid].get('files_changed', []):
                additions += file.get('additions', 0)
                deletions += file.get('deletions', 0)

            commit_summary[current_cid] = {
                'changed': files_changed,
                'additions': additions,
                'deletions': deletions,
                'count': len(current_cc_pair_df),
            }

            if prev_G is not None:
                added_vertices = set(G.nodes()) - set(prev_G.nodes())
                removed_vertices = set(prev_G.nodes()) - set(G.nodes())

                added_edges = set(G.edges()) - set(prev_G.edges())
                removed_edges = set(prev_G.edges()) - set(G.edges())

                vertex_change.append(added_vertices + removed_vertices)
                edge_change.append(added_edges + removed_edges)

            prev_G = G

        graph_stats = [vertex_change, edge_change]
        np.save(f"{project_name}_graph_stats", np.array(graph_stats))

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
  
        def linearize_metrics(metric_dict, metrics_order):
            linear_vectors = {}
            for cid, metrics in metric_dict.items():
                linear_vector = []
                for metric in metrics_order:
                    linear_vector.extend([metrics[metric][key] for key in sorted(metrics[metric].keys())])

                linear_vectors[cid] = linear_vector
            return linear_vectors
        
        def linearize_commit_summary(commit_summary):
            linearized_summary = {}
            for cid, metrics in commit_summary.items():
                linear_vector = [metrics['changed'], metrics['additions'], metrics['deletions'],  metrics['count']]
                linearized_summary[cid] = linear_vector
            return linearized_summary

        def attach_cid_to_features(metrics_dicts, metrics_order1, commit_ids):
            linearized_metrics = {}
            for name, metric_dict in metrics_dicts.items():
                if name != 'commit_summary':
                    linearized_metrics[name] = linearize_metrics(metric_dict, metrics_order1)
                else:
                    linearized_metrics[name] = linearize_commit_summary(metric_dict)

            consolidated_features = {}
            for cid in commit_ids:
                consolidated_features[cid] = {name: linearized_metrics[name].get(cid, []) for name in linearized_metrics}

            return consolidated_features

        metrics_dicts = {
            "global_graph_scores": global_graph_scores,
            "changed_graph_scores": changed_graph_scores,
            "global_graph_diffs": global_graph_diffs,
            "changed_graph_diffs": changed_graph_diffs,
            "commit_summary": commit_summary
        }

        return attach_cid_to_features(metrics_dicts, ['pagerank', 'centrality', 'closeness', 'clustering_coefficient'], commit_ids)

    
    def flatten_features(self, features_dict, y):
        flattened_lists = []
        for cid in features_dict:
            flattened_lists.append([])
            for feat_key in features_dict[cid]:
                flattened_lists[-1].extend(features_dict[cid][feat_key])

        
        max_length = max(len(lst) for lst in flattened_lists)
        filtered_flattened_lists = []
        filtered_y = []

        for i, lst in enumerate(flattened_lists):
            if len(lst) == max_length:
                filtered_flattened_lists.append(lst)
                filtered_y.append(y[i])

        x = np.array(filtered_flattened_lists)
        y = np.array(filtered_y)

        means = np.nanmean(x, axis=0)
        x = np.where(np.isnan(x), means, x)

        return x, y
    
    def create_graph_stats(self, project_name):
        filename = f"raw/{project_name}.json"
        commit_data = MockUtils.read_json(filename)

        graph_creator = GraphCreator()

        commit_ids = list(commit_data.keys())
        
        vertex_change = []
        edge_change = []
        prev_G = None
        for i in tqdm(range(len(commit_ids) - 1), "Calculating Graphs Stats"):
            current_cid = commit_ids[i]

            try:
                current_cc_pair_df = self.read_and_parse_csv(f'caller_callee_outputs/{project_name}/{current_cid}.csv', project_name)
            except FileNotFoundError:
                continue

            G = graph_creator.create_graph(current_cc_pair_df)

            if prev_G is not None:
                added_vertices = set(G.nodes()) - set(prev_G.nodes())
                removed_vertices = set(prev_G.nodes()) - set(G.nodes())

                added_edges = set(G.edges()) - set(prev_G.edges())
                removed_edges = set(prev_G.edges()) - set(G.edges())

                vertex_change.append(len(added_vertices.union(removed_vertices)))
                edge_change.append(len(added_edges.union(removed_edges)))

                if len(added_edges.union(removed_edges)) == 10:
                    print(current_cid)
                    print(added_edges)
                    print(removed_edges)
            else:
                vertex_change.append(0)
                edge_change.append(0)

            prev_G = G

        graph_stats = [vertex_change, edge_change]
        print(graph_stats)