import tempfile
import os
import shutil
from contextlib import contextmanager
import git  # For exc and other git module access
from git import Repo
from urllib.parse import urlparse
from typing import Dict, Optional, Tuple, TypedDict, Generator


class CloneResult(TypedDict):
    success: bool
    path: Optional[str]
    repo_name: Optional[str]
    msg: Optional[str]


@contextmanager
def clone_repo(
    url: str,
    branch: Optional[str] = None,
    depth: Optional[int] = None,
    recurse_submodules: bool = False
) -> Generator[Tuple[str, str], None, None]:
    """
    Clone a Git repository into a temporary directory and yield the path and repo name.
    Automatically cleans up the temporary directory after use.

    Args:
        url (str): The Git repository URL to clone.
        branch (Optional[str]): Specific branch to clone (default: None, uses default branch).
        depth (Optional[int]): Shallow clone depth (default: None, full clone).
        recurse_submodules (bool): If True, initialize and update submodules after cloning (default: False).

    Yields:
        Tuple[str, str]: (temp_path, repo_name)

    Raises:
        ValueError: If the URL is invalid or malformed.
        git.exc.GitCommandError: If the clone or submodule operation fails.
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
    repo = None
    try:
        tmp_path = tempfile.mkdtemp(prefix='codegen_')
        clone_options = {}
        if branch:
            clone_options['branch'] = branch
        if depth:
            clone_options['depth'] = depth

        repo = Repo.clone_from(url, tmp_path, **clone_options)

        if recurse_submodules:
            repo.git.submodule('update', '--init', '--recursive')

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


def clone(
    url: str,
    branch: Optional[str] = None,
    depth: Optional[int] = None,
    recurse_submodules: bool = False
) -> CloneResult:
    """
    Synchronous wrapper for cloning a Git repository.

    Args:
        url (str): The Git repository URL to clone.
        branch (Optional[str]): Specific branch to clone (default: None).
        depth (Optional[int]): Shallow clone depth (default: None).
        recurse_submodules (bool): If True, initialize and update submodules (default: False).

    Returns:
        CloneResult: {'success': bool, 'path': str or None, 'repo_name': str or None, 'msg': str or None}
    """
    try:
        with clone_repo(url, branch, depth, recurse_submodules) as (path, repo_name):
            return {'success': True, 'path': path, 'repo_name': repo_name, 'msg': None}
    except Exception as e:
        return {'success': False, 'path': None, 'repo_name': None, 'msg': str(e)}