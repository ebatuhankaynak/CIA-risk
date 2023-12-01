import requests
import json
from tqdm import tqdm

GITHUB_API_TOKEN = "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"

repo_names = {
    "Cli": "apache/commons-cli",
    "Closure": "google/closure-compiler",
    "Codec": "apache/commons-codec",
    "Compress": "apache/commons-compress",
    "Gson": "google/gson",
    "JacksonCore": "FasterXML/jackson-core",
    "Jsoup": "jhy/jsoup",
    "Lang": "apache/commons-lang",
    "Math": "apache/commons-math",
    "Mockito": "mockito/mockito",
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

def get_commit_details(owner, repo, sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {'Authorization': f'token {GITHUB_API_TOKEN}'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching commit details for {sha}")
        return None
    except requests.exceptions.SSLError:
        print("SSL Error on commit")
        return None

def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

for project_name, repo_path in repo_names.items():
    commit_data = {}
    owner, repo = repo_path.split('/')
    commits = get_commits(owner, repo)
    for commit in tqdm(commits):
        sha = commit['sha']
        details = get_commit_details(owner, repo, sha)
        print(details)
        commit_data[sha] = list(reversed(commits))

    save_to_json(commit_data, f"raw/{project_name}.json")
