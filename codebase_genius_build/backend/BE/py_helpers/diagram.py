import tempfile
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator
from urllib.parse import urlparse
import logging

from git import Repo
from git.exc import GitCommandError, GitError

# Configure logging
logger = logging.getLogger(__name__)


class GitCloneError(Exception):
    """Base exception for git clone operations."""
    pass


class InvalidURLError(GitCloneError):
    """Raised when the repository URL is invalid."""
    pass


class CloneResult:
    """Result object for clone operations with convenient access."""
    
    def __init__(
        self,
        success: bool,
        path: Optional[str] = None,
        repo_name: Optional[str] = None,
        msg: Optional[str] = None,
        branch: Optional[str] = None
    ):
        self.success = success
        self.path = path
        self.repo_name = repo_name
        self.msg = msg
        self.branch = branch
    
    def to_dict(self) -> dict[str, Optional[str | bool]]:
        """Convert to dictionary format."""
        return {
            'success': self.success,
            'path': self.path,
            'repo_name': self.repo_name,
            'msg': self.msg,
            'branch': self.branch
        }
    
    def __repr__(self) -> str:
        return f"CloneResult(success={self.success}, repo_name={self.repo_name})"


def extract_repo_name(url: str) -> str:
    """
    Extract repository name from a Git URL.
    
    Args:
        url: Git repository URL.
        
    Returns:
        Repository name without .git extension.
        
    Raises:
        InvalidURLError: If repository name cannot be extracted.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    
    # Handle both HTTP(S) and SSH URLs
    repo_name = os.path.basename(path)
    repo_name = repo_name.removesuffix('.git')
    
    if not repo_name or repo_name == '.':
        raise InvalidURLError(f"Could not extract repository name from URL: {url}")
    
    return repo_name


def validate_url(url: str) -> None:
    """
    Validate a Git repository URL.
    
    Args:
        url: Git repository URL to validate.
        
    Raises:
        InvalidURLError: If URL is invalid or malformed.
    """
    if not url or not url.strip():
        raise InvalidURLError("URL cannot be empty")
    
    url = url.strip()
    parsed = urlparse(url)
    
    # Check for common Git URL patterns
    is_ssh = url.startswith('git@') or parsed.scheme in ('ssh', 'git')
    is_http = parsed.scheme in ('http', 'https')
    is_file = parsed.scheme == 'file' or (not parsed.scheme and os.path.isabs(url))
    
    if not (is_ssh or is_http or is_file):
        raise InvalidURLError(
            f"Invalid Git URL format: {url}. "
            "Expected HTTP(S), SSH (git@...), or file path."
        )
    
    if is_http and not parsed.netloc:
        raise InvalidURLError(f"Invalid Git URL: missing hostname in {url}")


@contextmanager
def clone_repo(
    url: str,
    branch: Optional[str] = None,
    depth: Optional[int] = None,
    recurse_submodules: bool = False,
    single_branch: bool = True,
    timeout: Optional[int] = None
) -> Generator[tuple[str, str, str], None, None]:
    """
    Clone a Git repository into a temporary directory and yield details.
    Automatically cleans up the temporary directory after use.

    Args:
        url: The Git repository URL to clone.
        branch: Specific branch to clone (default: None, uses default branch).
        depth: Shallow clone depth (default: None, full clone).
        recurse_submodules: Initialize and update submodules after cloning (default: False).
        single_branch: Only clone the specified branch (default: True).
        timeout: Timeout in seconds for git operations (default: None).

    Yields:
        Tuple[str, str, str]: (temp_path, repo_name, active_branch)

    Raises:
        InvalidURLError: If the URL is invalid or malformed.
        GitCloneError: If the clone or submodule operation fails.
    """
    validate_url(url)
    
    repo_name = extract_repo_name(url)
    tmp_path: Optional[Path] = None
    repo: Optional[Repo] = None
    
    try:
        # Create temporary directory
        tmp_path = Path(tempfile.mkdtemp(prefix=f'gitclone_{repo_name}_'))
        logger.info(f"Cloning {url} into {tmp_path}")
        
        # Build clone options
        clone_kwargs = {
            'multi_options': []
        }
        
        if branch:
            clone_kwargs['branch'] = branch
        if depth and depth > 0:
            clone_kwargs['depth'] = depth
        if single_branch and branch:
            clone_kwargs['multi_options'].append('--single-branch')
        
        # Perform clone
        try:
            repo = Repo.clone_from(url, str(tmp_path), **clone_kwargs)
        except GitCommandError as e:
            raise GitCloneError(f"Git clone failed: {e.stderr or str(e)}") from e
        
        # Handle submodules if requested
        if recurse_submodules:
            try:
                logger.info("Initializing submodules")
                repo.git.submodule('update', '--init', '--recursive')
            except GitCommandError as e:
                logger.warning(f"Submodule initialization failed: {e}")
                # Don't fail the entire operation for submodule issues
        
        # Get active branch name
        try:
            active_branch = repo.active_branch.name
        except TypeError:
            # Detached HEAD state
            active_branch = str(repo.head.commit)[:8]
        
        logger.info(f"Successfully cloned {repo_name} (branch: {active_branch})")
        yield str(tmp_path), repo_name, active_branch
        
    except InvalidURLError:
        raise
    except GitError as e:
        raise GitCloneError(f"Git operation failed: {str(e)}") from e
    except Exception as e:
        raise GitCloneError(f"Unexpected error during clone: {str(e)}") from e
    finally:
        # Always cleanup
        if tmp_path and tmp_path.exists():
            try:
                shutil.rmtree(tmp_path, ignore_errors=True)
                logger.debug(f"Cleaned up temporary directory: {tmp_path}")
            except Exception as e:
                logger.error(f"Failed to cleanup {tmp_path}: {e}")


def clone(
    url: str,
    branch: Optional[str] = None,
    depth: Optional[int] = None,
    recurse_submodules: bool = False,
    single_branch: bool = True,
    timeout: Optional[int] = None
) -> CloneResult:
    """
    Clone a Git repository and return result (non-persistent).
    
    Note: This function clones and immediately cleans up. The cloned
    repository will not be available after this function returns.
    For persistent clones, use clone_persistent() or the clone_repo()
    context manager directly.

    Args:
        url: The Git repository URL to clone.
        branch: Specific branch to clone (default: None).
        depth: Shallow clone depth (default: None).
        recurse_submodules: Initialize and update submodules (default: False).
        single_branch: Only clone the specified branch (default: True).
        timeout: Timeout in seconds for git operations (default: None).

    Returns:
        CloneResult object with success status and details.
    """
    try:
        with clone_repo(
            url,
            branch=branch,
            depth=depth,
            recurse_submodules=recurse_submodules,
            single_branch=single_branch,
            timeout=timeout
        ) as (path, repo_name, active_branch):
            return CloneResult(
                success=True,
                path=path,
                repo_name=repo_name,
                branch=active_branch,
                msg=None
            )
    except GitCloneError as e:
        logger.error(f"Clone failed: {e}")
        return CloneResult(
            success=False,
            path=None,
            repo_name=None,
            msg=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return CloneResult(
            success=False,
            path=None,
            repo_name=None,
            msg=f"Unexpected error: {str(e)}"
        )


def clone_persistent(
    url: str,
    target_dir: Optional[str | Path] = None,
    branch: Optional[str] = None,
    depth: Optional[int] = None,
    recurse_submodules: bool = False,
    single_branch: bool = True
) -> CloneResult:
    """
    Clone a Git repository to a persistent location.

    Args:
        url: The Git repository URL to clone.
        target_dir: Directory to clone into (default: current directory).
        branch: Specific branch to clone (default: None).
        depth: Shallow clone depth (default: None).
        recurse_submodules: Initialize and update submodules (default: False).
        single_branch: Only clone the specified branch (default: True).

    Returns:
        CloneResult object with success status and details.
    """
    try:
        validate_url(url)
        repo_name = extract_repo_name(url)
        
        if target_dir:
            clone_path = Path(target_dir) / repo_name
        else:
            clone_path = Path.cwd() / repo_name
        
        if clone_path.exists():
            raise GitCloneError(f"Target directory already exists: {clone_path}")
        
        clone_kwargs = {'multi_options': []}
        if branch:
            clone_kwargs['branch'] = branch
        if depth and depth > 0:
            clone_kwargs['depth'] = depth
        if single_branch and branch:
            clone_kwargs['multi_options'].append('--single-branch')
        
        repo = Repo.clone_from(url, str(clone_path), **clone_kwargs)
        
        if recurse_submodules:
            repo.git.submodule('update', '--init', '--recursive')
        
        try:
            active_branch = repo.active_branch.name
        except TypeError:
            active_branch = str(repo.head.commit)[:8]
        
        return CloneResult(
            success=True,
            path=str(clone_path),
            repo_name=repo_name,
            branch=active_branch,
            msg=None
        )
        
    except GitCloneError as e:
        return CloneResult(success=False, msg=str(e))
    except Exception as e:
        return CloneResult(success=False, msg=f"Unexpected error: {str(e)}")


def main():
    """Example usage of git clone utilities."""
    # Example 1: Temporary clone with context manager
    try:
        with clone_repo('https://github.com/python/cpython', depth=1) as (path, name, branch):
            print(f"✓ Cloned {name} (branch: {branch}) to {path}")
            # Work with repository here
            # Automatically cleaned up after this block
    except GitCloneError as e:
        print(f"✗ Clone failed: {e}")
    
    # Example 2: Non-persistent clone (for inspection only)
    result = clone('https://github.com/python/cpython', depth=1)
    print(f"\n{'✓' if result.success else '✗'} Clone result: {result}")


if __name__ == '__main__':
    main()