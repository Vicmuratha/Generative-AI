import tempfile
import os
import shutil
from contextlib import contextmanager
import git
from git import Repo
from urllib.parse import urlparse
from typing import Dict, Optional, Tuple, Generator


@contextmanager
def clone_repo(url: str, branch: Optional[str] = None, depth: Optional[int] = None) -> Generator[Tuple[str, str], None, None]:
    """
    Clone a Git repository into a temporary directory and yield the path and repo name.
    Automatically cleans up the temporary directory after use.

    Args:
        url (str): The Git repository URL to clone.
        branch (Optional[str]): Specific branch to clone (default: None, uses default branch).
        depth (Optional[int]): Shallow clone depth (default: None, full clone).

    Yields:
        Tuple[str, str]: (temp_path, repo_name)

    Raises:
        ValueError: If the URL is invalid or malformed.
        git.exc.GitCommandError: If the clone operation fails.
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty.")

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Git URL: {url}")

    repo_name = os.path.basename(parsed.path.rstrip('/')).rstrip('.git')
    if not repo_name:
        raise ValueError(f"Could not extract repository name from URL: {url}")

    tmp_path = None
    try:
        tmp_path = tempfile.mkdtemp(prefix='codegen_')
        clone_options = {}
        if branch:
            clone_options['branch'] = branch
        if depth:
            clone_options['depth'] = depth

        Repo.clone_from(url, tmp_path, **clone_options)
        yield tmp_path, repo_name
    except Exception as e:
        if tmp_path:
            shutil.rmtree(tmp_path, ignore_errors=True)
        if isinstance(e, (ValueError, git.exc.GitCommandError)):
            raise
        raise RuntimeError(f"Failed to clone repository: {str(e)}") from e
    finally:
        if tmp_path:
            shutil.rmtree(tmp_path, ignore_errors=True)


def clone(url: str, branch: Optional[str] = None, depth: Optional[int] = None) -> Dict[str, Optional[str]]:
    """
    Legacy synchronous wrapper for cloning a Git repository.

    Args:
        url (str): The Git repository URL to clone.
        branch (Optional[str]): Specific branch to clone (default: None).
        depth (Optional[int]): Shallow clone depth (default: None).

    Returns:
        Dict[str, Optional[str]]: {'success': bool, 'path': str or None, 'repo_name': str or None, 'msg': str or None}
    """
    try:
        with clone_repo(url, branch, depth) as (path, repo_name):
            return {'success': True, 'path': path, 'repo_name': repo_name, 'msg': None}
    except Exception as e:
        return {'success': False, 'path': None, 'repo_name': None, 'msg': str(e)}