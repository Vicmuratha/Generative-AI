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
    depth: Optional[int] = None
) -> Generator[tuple[str, str], None, None]:
    """
    Clone a Git repository into a temporary directory and yield details.
    Automatically cleans up the temporary directory after use.

    Args:
        url: The Git repository URL to clone.
        branch: Specific branch to clone (default: None, uses default branch).
        depth: Shallow clone depth (default: None, full clone).

    Yields:
        Tuple[str, str]: (temp_path, repo_name)

    Raises:
        InvalidURLError: If the URL is invalid or malformed.
        GitCloneError: If the clone operation fails.
    """
    validate_url(url)
    
    repo_name = extract_repo_name(url)
    tmp_path: Optional[Path] = None
    
    try:
        # Create temporary directory
        tmp_path = Path(tempfile.mkdtemp(prefix=f'codegen_{repo_name}_'))
        logger.info(f"Cloning {url} into {tmp_path}")
        
        # Build clone options
        clone_kwargs = {}
        if branch:
            clone_kwargs['branch'] = branch
        if depth and depth > 0:
            clone_kwargs['depth'] = depth
        
        # Perform clone
        try:
            Repo.clone_from(url, str(tmp_path), **clone_kwargs)
        except GitCommandError as e:
            raise GitCloneError(f"Git clone failed: {e.stderr or str(e)}") from e
        
        logger.info(f"Successfully cloned {repo_name}")
        yield str(tmp_path), repo_name
        
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
    depth: Optional[int] = None
) -> dict[str, Optional[str | bool]]:
    """
    Legacy synchronous wrapper for cloning a Git repository.
    
    Note: This function clones and immediately cleans up. The cloned
    repository will not be available after this function returns.
    Consider using the clone_repo() context manager for working with
    the cloned repository.

    Args:
        url: The Git repository URL to clone.
        branch: Specific branch to clone (default: None).
        depth: Shallow clone depth (default: None).

    Returns:
        Dict with keys: 'success' (bool), 'path' (str|None), 
        'repo_name' (str|None), 'msg' (str|None)
    """
    try:
        with clone_repo(url, branch=branch, depth=depth) as (path, repo_name):
            return {
                'success': True,
                'path': path,
                'repo_name': repo_name,
                'msg': None
            }
    except GitCloneError as e:
        logger.error(f"Clone failed: {e}")
        return {
            'success': False,
            'path': None,
            'repo_name': None,
            'msg': str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            'success': False,
            'path': None,
            'repo_name': None,
            'msg': f"Unexpected error: {str(e)}"
        }


def main():
    """Example usage of git clone utilities."""
    # Example 1: Using context manager (recommended)
    print("Example 1: Using context manager")
    try:
        with clone_repo('https://github.com/psf/requests', depth=1) as (path, name):
            print(f"✓ Cloned {name} to {path}")
            # Work with repository here
            print(f"  Contents: {os.listdir(path)[:5]}...")
            # Automatically cleaned up after this block
    except GitCloneError as e:
        print(f"✗ Clone failed: {e}")
    
    # Example 2: Using legacy function
    print("\nExample 2: Using legacy clone() function")
    result = clone('https://github.com/psf/requests', depth=1)
    if result['success']:
        print(f"✓ Clone succeeded: {result['repo_name']}")
        print(f"  Note: Path {result['path']} is already cleaned up!")
    else:
        print(f"✗ Clone failed: {result['msg']}")


if __name__ == '__main__':
    main()