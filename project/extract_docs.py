import io
import os
import json
import hashlib
import zipfile
import requests
import frontmatter

cache_dir = "./repo_cache"

def calc_repo_cache_key(repo_owner, repo_name):
    key = f"{repo_owner}/{repo_name}"
    return hashlib.sha256(key.encode()).hexdigest()

def get_cache_path(repo_owner, repo_name):
    cache_key = calc_repo_cache_key(repo_owner, repo_name)
    return os.path.join(cache_dir, f"{cache_key}.json")

def read_from_cache(repo_owner, repo_name):
    cache_file = get_cache_path(repo_owner, repo_name)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None

def write_to_cache(repo_owner, repo_name, data):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = get_cache_path(repo_owner, repo_name)
    with open(cache_file, "w") as f:
        json.dump(data, f)


def read_repo_data(repo_owner, repo_name, force_refresh=False):
    """
    Download and parse all markdown files from a GitHub repository.
    
    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name
        force_refresh: Whether to ignore cached data
    
    Returns:
        List of dictionaries containing file content and metadata
    """
    if not force_refresh:
        cached_data = read_from_cache(repo_owner, repo_name)
        if cached_data:
            return cached_data

    prefix = 'https://codeload.github.com' 
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for file_info in zf.infolist():
            filename = file_info.filename
            filename_lower = filename.lower()

            if not (filename_lower.endswith('.md') 
                or filename_lower.endswith('.mdx')):
                continue

            try:
                with zf.open(file_info) as f_in:
                    content = f_in.read().decode('utf-8', errors='ignore')
                    post = frontmatter.loads(content)
                    data = post.to_dict()
                    data['filename'] = filename
                    repository_data.append(data)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    write_to_cache(repo_owner, repo_name, repository_data)

    return repository_data


if __name__ == "__main__":
    print("Extracting documents for Github Copilot Chat...")
    copilot_chat_docs = read_repo_data('microsoft', 'vscode-copilot-chat')
    print(f"Github Copilot Chat documents: {len(copilot_chat_docs)}")
