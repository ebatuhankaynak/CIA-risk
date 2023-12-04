import pandas as pd
import json

fonte_dataset_path = "fonte_dataset.csv"
fonte_dataset = pd.read_csv(fonte_dataset_path)

project_name = "Time"
raw_json_path = f"raw/{project_name}.json"
with open(raw_json_path, 'r') as file:
    data = json.load(file)

commit_ids = set(data.keys())
commit_ids = {sha[:7] for sha in commit_ids}
print(commit_ids)
fonte_dataset_for_pid = fonte_dataset[fonte_dataset['pid'] == project_name]
print(fonte_dataset_for_pid)

fonte_commit_ids = set(fonte_dataset_for_pid['commit'])
labels = [1 if commit_id in fonte_commit_ids else 0 for commit_id in commit_ids]

