import base64
import requests
from fastapi import HTTPException
from app.core.config import settings

def extract_repo_url(user_input: str) -> str:
    words = user_input.split()
    for word in words:
        if "github.com" in word:
            return word
    return None


def extract_user_and_repo(repo_url: str):
    try:
        parts = repo_url.strip().rstrip("/").split("/")
        user = parts[-2]
        repo = parts[-1]
        return user, repo
    except IndexError:
        raise ValueError("Invalid GitHub repository URL")


def get_file_content(file_url: str, headers: dict, file_type: str):
    ALLOWED_FILE_TYPES = ["py", "md", "ipynb", "html"]
    response = requests.get(file_url, headers=headers)
    if response.status_code != 200:
        error_message = response.json().get("message", "Unknown error")
        raise HTTPException(
            status_code=response.status_code, detail={"message": error_message}
        )

    ext_idx = file_type.rfind(".")
    file_ext = file_type[ext_idx + 1 :]

    if file_ext in ALLOWED_FILE_TYPES:
        file_content = response.json().get("content")
        if not file_content:
            return ""

        # decode from base64
        decoded_file = ""
        try:
            decoded_file = base64.b64decode(file_content).decode("utf-8")
        except Exception as error:
            raise HTTPException(status_code=500, detail={"error": error})

        return decoded_file
    else:
        return ""


def get_dir_content(dir_url: str, headers: dict):
    EXCLUDE_DIRS = ["venv", "node_modules", "__pycache__"]
    response = requests.get(dir_url, headers=headers)
    if response.status_code != 200:
        error_message = response.json().get("message", "Unknown error")
        raise HTTPException(
            status_code=response.status_code, detail={"message": error_message}
        )

    repo_contents = []
    for item in response.json():
        if item["type"] == "file":
            file_content = get_file_content(item["url"], headers, item["name"])
            # append if file content is not empty
            if file_content:
                repo_contents.append({"name": item["name"], "content": file_content})
                # repo_contents.append({"type": "text", "text": file_content})
        elif item["type"] == "dir" and item["name"] not in EXCLUDE_DIRS:
            dir_content = get_dir_content(item["url"], headers)
            # append if there are valid files inside the dir
            if len(dir_content):
                repo_contents.extend(dir_content)

    return repo_contents


def get_github_repo_contents(owner: str, repo: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token != "":
        headers["Authorization"] = f"Bearer {settings.github_token}"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        error_message = response.json().get("message", "Unknown error")
        raise HTTPException(
            status_code=response.status_code, detail={"message": error_message}
        )

    # recursive logic for reading all files in the repo
    repo_contents = []
    for item in response.json():
        if item["type"] == "file":
            file_content = get_file_content(item["url"], headers, item["name"])
            if file_content:
                repo_contents.append({"name": item["name"], "content": file_content})
                # repo_contents.append({"type": "text", "text": file_content})
        elif item["type"] == "dir":
            dir_content = get_dir_content(item["url"], headers)
            if dir_content:
                repo_contents.extend(dir_content)

    return repo_contents
