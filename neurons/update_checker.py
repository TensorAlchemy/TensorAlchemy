import subprocess

import httpx
from loguru import logger

from neurons.utils.common import show_warning_message


def get_current_branch():
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_local_commit_hash(branch):
    result = subprocess.run(
        ["git", "rev-parse", branch], capture_output=True, text=True
    )
    return result.stdout.strip()


def get_remote_commit_hash(repo_url, branch):
    api_url = f"https://api.github.com/repos/{repo_url}/commits/{branch}"
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(api_url)
            response.raise_for_status()

        return response.json()["sha"]

    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP error occurred: "
            + f"{e.response.status_code} {e.response.reason_phrase}"
        )

    except httpx.RequestError as e:
        logger.error(
            "An error occurred while requesting " + e.request.url,
        )
    return None


def show_update_warning_message(local_commit, remote_commit):
    msg = f"""
    Your TensorAlchemy is OUTDATED

    Your local TensorAlchemy is not up-to-date with
    the TensorAlchemy repository.
    Your hash:   {local_commit}
    Remote hash: {remote_commit}

    Please update:
    1) git fetch && git reset --hard origin/main
    2) Restart your validator
    """.strip()
    show_warning_message(msg)


def check_for_updates() -> None:
    repo_url: str = "TensorAlchemy/TensorAlchemy"

    current_branch = get_current_branch()

    local_commit = get_local_commit_hash(current_branch)
    remote_commit = get_remote_commit_hash(repo_url, current_branch)

    if local_commit and remote_commit and local_commit != remote_commit:
        show_update_warning_message(local_commit, remote_commit)

    elif local_commit == remote_commit:
        logger.info(
            "Your local repository is up-to-date with the GitHub repository."
        )
    else:
        logger.info("Unable to determine the update status.")


def safely_check_for_updates():
    try:
        check_for_updates()
    except Exception as e:
        logger.error(f"Failed to check for updates {e}")
