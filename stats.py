import json

def read_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def get_total_authors_and_commits(commit_data):
    authors = set()
    for commit in commit_data.values():
        authors.add(commit['author_name'])
    total_commits = len(commit_data)
    return len(authors), total_commits

def get_commit_info(commit_data, sha):
    commit_info = commit_data.get(sha)
    if commit_info:
        changed_files = [file['filename'] for file in commit_info['files_changed']]
        return changed_files
    return None

def get_project_files(commit_data, sha):
    files = {}
    for commit_sha, commit in commit_data.items():
        for file in commit['files_changed']:
            if file['status'] in ['added', 'modified']:
                files[file['filename']] = True
            elif file['status'] == 'deleted':
                files.pop(file['filename'], None)
        if commit_sha == sha:
            break
    return list(files.keys())

filename = "raw/Cli.json"
commit_data = read_json(filename)

keys = list(commit_data.keys())
sha = keys[len(keys) // 2]
changed_files = get_commit_info(commit_data, sha)
if changed_files is not None:

    # Print total list of project files up to that commit
    project_files = get_project_files(commit_data, sha)
    print(f"Total Project Files as of Commit {sha}:")
    for file in project_files:
        print(f" - {file}")

    print(f"Changed Files in Commit {sha}:")
    for file in changed_files:
        print(f" - {file}")
else:
    print(f"No commit found for SHA {sha}")

total_authors, total_commits = get_total_authors_and_commits(commit_data)
print(f"Total Authors: {total_authors}")
print(f"Total Commits: {total_commits}")