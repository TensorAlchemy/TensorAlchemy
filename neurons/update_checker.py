import subprocess

import requests
from loguru import logger


def get_current_branch():
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip()


def get_local_commit_hash(branch):
    result = subprocess.run(
        ["git", "rev-parse", branch], capture_output=True, text=True
    )
    return result.stdout.strip()


def get_remote_commit_hash(repo_url, branch):
    api_url = f"https://api.github.com/repos/{repo_url}/commits/{branch}"
    response = requests.get(api_url, timeout=10)
    if response.status_code == 200:
        return response.json()["sha"]
    else:
        print("Failed to fetch remote commit hash")
        return None


def show_warning_message(local_commit, remote_commit):
    window_width = 60

    def create_line(content):
        padding = window_width - len(content) - 4
        return f"* {content}{' ' * padding} *"

    line = "*" * window_width
    new_line = f"\n{line}"
    empty_line = create_line("")
    warning_lines = [
        create_line("WARNING: Your TensorAlchemy OUTDATED"),
        create_line("Your local TensorAlchemy is not up-to-date with"),
        create_line("the TensorAlchemy repository."),
        create_line(f"Your hash  : {local_commit}"),
        create_line(f"Remote hash: {remote_commit}"),
    ]

    message = "\n".join([new_line, empty_line] + warning_lines + [empty_line, line])

    logger.warning(message)


def check_for_updates(local_repo_path, repo_url):
    current_branch = get_current_branch()

    local_commit = get_local_commit_hash(current_branch)
    remote_commit = get_remote_commit_hash(repo_url, current_branch)

    if local_commit and remote_commit and local_commit != remote_commit:
        show_warning_message(local_commit, remote_commit)
    elif local_commit == remote_commit:
        logger.info("Your local repository is up-to-date with the GitHub repository.")
    else:
        logger.info("Unable to determine the update status.")
