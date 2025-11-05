import os
from pathlib import Path
from typing import List, Dict, Optional


IGNORED_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env'}  # Expanded common ignored dirs
FILE_EXTENSIONS = {'.py', '.jac', '.md'}  # Set for faster lookup


def build(path: str | Path) -> Dict[str, List[str]]:
    """
    Walk a directory and collect relative paths of files with specified extensions,
    ignoring certain directories.

    Args:
        path (str | Path): Root directory path to scan.

    Returns:
        Dict[str, List[str]]: {'files': list of relative file paths (forward slashes)}
    """
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"Path '{path}' is not a directory.")

    files: List[str] = []
    for root, dirs, filenames in os.walk(path):
        # Modify dirs in-place to skip ignored subdirs
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        # Collect matching files
        root_path = Path(root)
        for filename in filenames:
            if Path(filename).suffix in FILE_EXTENSIONS:
                rel_path = (root_path / filename).relative_to(path).as_posix()
                files.append(rel_path)

    return {'files': files}


def readme_summary(path: str | Path, max_length: int = 1000) -> str:
    """
    Read and truncate the content of README.md at the root path.

    Args:
        path (str | Path): Root directory path.
        max_length (int): Maximum characters to read (default: 1000).

    Returns:
        str: Truncated README content or empty string if not found.
    """
    path = Path(path).resolve()
    readme_path = path / 'README.md'
    if not readme_path.is_file():
        return ''

    try:
        with readme_path.open('r', encoding='utf-8') as f:
            content = f.read(max_length)
            # Truncate at sentence boundary if possible, for better readability
            if len(content) == max_length:
                last_period = content.rfind('.')
                if 50 < last_period < max_length - 10:  # Reasonable sentence cut
                    content = content[:last_period + 1]
            return content
    except (UnicodeDecodeError, IOError) as e:
        # Log or handle as needed; for now, return empty
        return ''


def makedirs(path: str | Path, exist_ok: bool = True) -> None:
    """
    Create directory at the given path, including parents.

    Args:
        path (str | Path): Directory path to create.
        exist_ok (bool): If False, raise if directory exists (default: True).
    """
    Path(path).mkdir(parents=True, exist_ok=exist_ok)


def write_text(path: str | Path, text: str, encoding: str = 'utf-8') -> None:
    """
    Write text to a file at the given path, creating directories if needed.

    Args:
        path (str | Path): File path to write.
        text (str): Content to write.
        encoding (str): File encoding (default: 'utf-8').

    Raises:
        IOError: If writing fails.
    """
    path = Path(path)
    makedirs(path.parent, exist_ok=True)
    try:
        with path.open('w', encoding=encoding) as f:
            f.write(text)
    except IOError as e:
        raise IOError(f"Failed to write to '{path}': {e}")