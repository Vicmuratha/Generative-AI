import os
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
IGNORED_DIRS = frozenset({
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env',
    'dist', 'build', '.pytest_cache', '.mypy_cache', 'htmlcov', '.tox'
})
FILE_EXTENSIONS = frozenset({'.py', '.jac', '.md'})


class FileUtilsError(Exception):
    """Base exception for file utility errors."""
    pass


class DirectoryNotFoundError(FileUtilsError):
    """Raised when a directory doesn't exist."""
    pass


def build(
    path: str | Path,
    extensions: Optional[frozenset[str]] = None,
    ignored_dirs: Optional[frozenset[str]] = None,
    follow_symlinks: bool = False
) -> dict[str, list[str]]:
    """
    Walk a directory and collect relative paths of files with specified extensions,
    ignoring certain directories.

    Args:
        path: Root directory path to scan.
        extensions: File extensions to include (default: FILE_EXTENSIONS).
        ignored_dirs: Directory names to skip (default: IGNORED_DIRS).
        follow_symlinks: Whether to follow symbolic links (default: False).

    Returns:
        Dictionary with 'files' key containing sorted list of relative file paths.

    Raises:
        DirectoryNotFoundError: If path doesn't exist or isn't a directory.
        PermissionError: If access to directory is denied.
    """
    path = Path(path).resolve()
    
    if not path.exists():
        raise DirectoryNotFoundError(f"Path '{path}' does not exist")
    if not path.is_dir():
        raise DirectoryNotFoundError(f"Path '{path}' is not a directory")

    extensions = extensions or FILE_EXTENSIONS
    ignored_dirs = ignored_dirs or IGNORED_DIRS
    files: list[str] = []

    try:
        for root, dirs, filenames in os.walk(path, followlinks=follow_symlinks):
            # Skip ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignored_dirs]

            root_path = Path(root)
            for filename in filenames:
                if Path(filename).suffix.lower() in extensions:
                    try:
                        file_path = root_path / filename
                        rel_path = file_path.relative_to(path).as_posix()
                        files.append(rel_path)
                    except ValueError:
                        # Skip files that can't be made relative (shouldn't happen)
                        logger.warning(f"Skipping file with invalid path: {filename}")
                        continue

    except PermissionError as e:
        logger.error(f"Permission denied accessing directory: {e}")
        raise

    return {'files': sorted(files)}


def readme_summary(
    path: str | Path,
    max_length: int = 1000,
    readme_name: str = 'README.md'
) -> str:
    """
    Read and truncate the content of a README file at the root path.

    Args:
        path: Root directory path.
        max_length: Maximum characters to read (default: 1000).
        readme_name: Name of README file (default: 'README.md').

    Returns:
        Truncated README content or empty string if not found/readable.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be positive, got {max_length}")

    path = Path(path).resolve()
    readme_path = path / readme_name

    if not readme_path.is_file():
        logger.debug(f"README not found at '{readme_path}'")
        return ''

    try:
        with readme_path.open('r', encoding='utf-8', errors='replace') as f:
            content = f.read(max_length)
            
            # Truncate at sentence boundary for better readability
            if len(content) == max_length:
                # Look for sentence endings
                for delimiter in ('.', '!', '?'):
                    last_delimiter = content.rfind(delimiter)
                    if 50 < last_delimiter < max_length - 10:
                        content = content[:last_delimiter + 1]
                        break
                else:
                    # No good sentence boundary found, add ellipsis
                    content = content.rstrip() + '...'
            
            return content.strip()

    except (UnicodeDecodeError, IOError) as e:
        logger.warning(f"Failed to read README at '{readme_path}': {e}")
        return ''


def makedirs(path: str | Path, exist_ok: bool = True, mode: int = 0o777) -> Path:
    """
    Create directory at the given path, including parent directories.

    Args:
        path: Directory path to create.
        exist_ok: If False, raise FileExistsError if directory exists (default: True).
        mode: Directory permissions (default: 0o777).

    Returns:
        Path object of the created directory.

    Raises:
        FileExistsError: If directory exists and exist_ok is False.
        PermissionError: If lacking permissions to create directory.
    """
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=exist_ok, mode=mode)
    return path


def write_text(
    path: str | Path,
    text: str,
    encoding: str = 'utf-8',
    create_dirs: bool = True,
    newline: Optional[str] = None
) -> Path:
    """
    Write text to a file, optionally creating parent directories.

    Args:
        path: File path to write.
        text: Content to write.
        encoding: File encoding (default: 'utf-8').
        create_dirs: Create parent directories if needed (default: True).
        newline: Newline character(s) to use (default: None for platform default).

    Returns:
        Path object of the written file.

    Raises:
        FileUtilsError: If writing fails.
        PermissionError: If lacking permissions to write.
    """
    path = Path(path).resolve()
    
    if create_dirs and not path.parent.exists():
        try:
            makedirs(path.parent, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise FileUtilsError(f"Failed to create parent directories for '{path}': {e}") from e

    try:
        with path.open('w', encoding=encoding, newline=newline) as f:
            f.write(text)
        logger.debug(f"Successfully wrote {len(text)} characters to '{path}'")
        return path

    except (IOError, OSError) as e:
        raise FileUtilsError(f"Failed to write to '{path}': {e}") from e


def read_text(
    path: str | Path,
    encoding: str = 'utf-8',
    errors: str = 'strict'
) -> str:
    """
    Read text from a file.

    Args:
        path: File path to read.
        encoding: File encoding (default: 'utf-8').
        errors: How to handle encoding errors (default: 'strict').

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        FileUtilsError: If reading fails.
    """
    path = Path(path).resolve()
    
    if not path.is_file():
        raise FileNotFoundError(f"File not found: '{path}'")

    try:
        with path.open('r', encoding=encoding, errors=errors) as f:
            return f.read()
    except (IOError, OSError) as e:
        raise FileUtilsError(f"Failed to read from '{path}': {e}") from e


def main():
    """Example usage of file utilities."""
    # Example: Build file list
    try:
        result = build('.', extensions=frozenset({'.py', '.md'}))
        print(f"Found {len(result['files'])} files:")
        for file in result['files'][:5]:  # Show first 5
            print(f"  - {file}")
        if len(result['files']) > 5:
            print(f"  ... and {len(result['files']) - 5} more")
    except FileUtilsError as e:
        print(f"Error: {e}")

    # Example: Read README
    summary = readme_summary('.', max_length=200)
    if summary:
        print(f"\nREADME summary:\n{summary}")


if __name__ == '__main__':
    main()