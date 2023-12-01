import requests
import json
import threading
from pqdm.processes import pqdm

GITHUB_API_TOKEN = "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"

repo_names = {
    "Cli": "apache/commons-cli",
    #"Closure": "google/closure-compiler",
    "Codec": "apache/commons-codec",
    #"Compress": "apache/commons-compress",
    "Gson": "google/gson",
    "JacksonCore": "FasterXML/jackson-core",
    "Jsoup": "jhy/jsoup",
    #"Lang": "apache/commons-lang",
    #"Math": "apache/commons-math",
    #"Mockito": "mockito/mockito",
    "Time": "JodaOrg/joda-time"
}

def get_commits(owner, repo):
    commits = []
    page = 1
    per_page = 100  # Maximum allowed by GitHub
    while True:
        print(f"Collecting page {page} for {owner}/{repo}")
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?page={page}&per_page={per_page}"
        headers = {'Authorization': f'token {GITHUB_API_TOKEN}'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            new_commits = response.json()
            if not new_commits:
                break
            commits.extend(new_commits)
            page += 1
        else:
            print(f"Error fetching commits for {owner}/{repo}: {response.status_code}")
            break
    return commits

def get_commit_details_task(args):
    owner, repo, sha, index = args
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {'Authorization': f'token {GITHUB_API_TOKEN}'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return index, response.json()
        else:
            print(f"Error fetching commit details for {sha}")
        return index, None
    except requests.exceptions.SSLError:
        print("SSL Error on commit")
        return index, None

def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)


project_name = "Jsoup"
repo_path = repo_names[project_name]

commit_data = {}
owner, repo = repo_path.split('/')
commits = get_commits(owner, repo)

args_list = [(owner, repo, commit['sha'], index) for index, commit in enumerate(reversed(commits))]

commit_details_with_index  = pqdm(args_list, get_commit_details_task, n_jobs=10)
commit_details_with_index.sort(key=lambda x: x[0])
sorted_commit_details = [details for _, details in sorted(commit_details_with_index, key=lambda x: x[0])]

commit_data = {}
for details in sorted_commit_details:
    if details:
        sha = details['sha']
        commit_info = {
            "author_name": details['commit']['author']['name'],
            "files_changed": [
                {
                    "filename": file['filename'],
                    "status": file['status'],
                    "additions": file['additions'],
                    "deletions": file['deletions']
                }
                for file in details['files']
            ]
        }
        commit_data[sha] = commit_info

save_to_json(commit_data, f"raw/{project_name}.json")