"""Look up user information from local git."""
import subprocess
from typing import Optional


def get_git_user_name() -> Optional[str]:
    """Get the user's name from git, if it is configured, otherwise None."""
    try:
        return (
            subprocess.run(["git", "config", "--get", "user.name"], capture_output=True)
            .stdout.decode()
            .strip()
        )
    except FileNotFoundError:
        return None


def get_git_user_email() -> Optional[str]:
    """Get the user's email from git if it is configured, otherwise None."""
    try:
        return (
            subprocess.run(
                ["git", "config", "--get", "user.email"], capture_output=True
            )
            .stdout.decode()
            .strip()
        )
    except FileNotFoundError:
        return None
