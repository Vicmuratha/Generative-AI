# repo_mapper.jac - Enhanced repository mapping with robust error handling
import:py from jaclang.core.py_module { clone, filetree_build, readme_summary, cleanup_directory }

# Configuration for repository operations
glob REPO_CONFIG: dict = {
    "max_repo_size_mb": 500,
    "clone_timeout_seconds": 300,
    "allowed_extensions": [".py", ".js", ".java", ".go", ".rs", ".cpp", ".c", ".jac"],
    "ignored_dirs": ["node_modules", ".git", "__pycache__", "venv", "env", "dist", "build"]
}

# Result status enum for consistency
enum CloneStatus {
    SUCCESS = "success",
    ERROR = "error",
    TIMEOUT = "timeout",
    TOO_LARGE = "too_large",
    INVALID_URL = "invalid_url"
}

# ============================================================================
# Main Clone and Map Walker
# ============================================================================

walker clone_and_map(url: str) -> dict {
    """
    Clones a repository and maps its structure.
    
    This walker handles the complete repository cloning process including:
    - URL validation
    - Repository cloning
    - File tree generation
    - README extraction and summarization
    
    Args:
        url: Repository URL (must be from supported hosting service)
        
    Returns:
        dict: Standardized result with structure:
            {
                "success": bool,
                "status": CloneStatus,
                "msg": str,
                "data": {
                    "repo_name": str,
                    "path": str,
                    "file_tree": dict,
                    "readme_summary": str,
                    "metadata": dict
                } | None,
                "errors": list,
                "timestamp": str
            }
    
    Example:
        result = clone_and_map("https://github.com/user/repo");
        if result["success"] {
            process_files(result["data"]["file_tree"]);
        }
    """
    
    # Initialize result structure
    result: dict = {
        "success": False,
        "status": CloneStatus.ERROR,
        "msg": "",
        "data": None,
        "errors": [],
        "timestamp": _get_timestamp()
    }
    
    # Step 1: Validate URL
    validation = _validate_repo_url(url)
    if not validation["valid"] {
        result["msg"] = "URL validation failed: " + validation["msg"]
        result["status"] = CloneStatus.INVALID_URL
        result["errors"].append({
            "step": "validation",
            "error": validation["msg"]
        })
        return result
    }
    
    # Step 2: Clone repository
    try {
        r = clone(url)
        
        if not r {
            result["msg"] = "Clone operation returned null"
            result["errors"].append({
                "step": "cloning",
                "error": "Null response from clone operation"
            })
            return result
        }
        
        if not r.success {
            error_msg = r.msg if hasattr(r, 'msg') and r.msg else "Unknown error"
            result["msg"] = "Repository cloning failed: " + error_msg
            result["status"] = _determine_clone_status(r)
            result["errors"].append({
                "step": "cloning",
                "error": error_msg
            })
            return result
        }
        
        # Validate clone result has required fields
        if not hasattr(r, 'path') or not r.path or not hasattr(r, 'repo_name') or not r.repo_name {
            result["msg"] = "Clone succeeded but missing required data (path or repo_name)"
            result["errors"].append({
                "step": "cloning",
                "error": "Missing required fields in clone result"
            })
            return result
        }
        
    } except Exception as e {
        result["msg"] = "Clone operation threw exception: " + str(e)
        result["errors"].append({
            "step": "cloning",
            "error": str(e)
        })
        return result
    }
    
    # Step 3: Build file tree
    ft: dict | None = None
    try {
        ft = filetree_build(r.path)
        
        if not ft {
            result["errors"].append({
                "step": "file_tree",
                "error": "File tree generation returned null"
            })
            ft = {"files": [], "directories": []}
        }
        
        # Validate file tree structure
        if not _validate_file_tree(ft) {
            result["errors"].append({
                "step": "file_tree",
                "error": "Invalid file tree structure"
            })
            ft = {"files": [], "directories": []}
        }
        
    } except Exception as e {
        result["errors"].append({
            "step": "file_tree",
            "error": str(e)
        })
        ft = {"files": [], "directories": []}
    }
    
    # Step 4: Extract and summarize README
    rs: str = "No README summary available"
    try {
        readme_result = readme_summary(r.path)
        
        if readme_result {
            rs = readme_result
        } else {
            result["errors"].append({
                "step": "readme",
                "error": "README summary returned null"
            })
        }
        
    } except Exception as e {
        result["errors"].append({
            "step": "readme",
            "error": str(e)
        })
    }
    
    # Step 5: Generate metadata
    metadata = _generate_metadata(r, ft, rs)
    
    # Build successful result
    result["success"] = True
    result["status"] = CloneStatus.SUCCESS
    result["msg"] = "Repository mapped successfully" + (
        " with warnings" if len(result["errors"]) > 0 else ""
    )
    result["data"] = {
        "repo_name": r.repo_name,
        "path": r.path,
        "file_tree": ft,
        "readme_summary": rs,
        "metadata": metadata
    }
    
    return result
}

# ============================================================================
# Validation Helpers
# ============================================================================

can _validate_repo_url(url: str) -> dict {
    """
    Validates repository URL format and source.
    
    Args:
        url: Repository URL to validate
        
    Returns:
        dict: {"valid": bool, "msg": str, "source": str|None}
    """
    
    result: dict = {
        "valid": False,
        "msg": "",
        "source": None
    }
    
    # Check for empty URL
    if not url or len(url) == 0 {
        result["msg"] = "URL cannot be empty"
        return result
    }
    
    # Trim whitespace
    url = url.strip()
    
    # Check for basic URL structure
    if not url.startswith("http://") and not url.startswith("https://") {
        result["msg"] = "URL must start with http:// or https://"
        return result
    }
    
    # Check for supported hosting services
    supported_sources: dict = {
        "github.com": "GitHub",
        "gitlab.com": "GitLab",
        "bitbucket.org": "Bitbucket",
        "codeberg.org": "Codeberg"
    }
    
    source_found: bool = False
    for domain in supported_sources {
        if domain in url {
            source_found = True
            result["source"] = supported_sources[domain]
            break
        }
    }
    
    if not source_found {
        result["msg"] = "URL must be from a supported source (GitHub, GitLab, Bitbucket, or Codeberg)"
        return result
    }
    
    # Additional validation: Check for reasonable URL structure
    if url.count("/") < 3 {
        result["msg"] = "URL appears to be incomplete (missing repository path)"
        return result
    }
    
    result["valid"] = True
    result["msg"] = "Valid " + result["source"] + " repository URL"
    return result
}

can _validate_file_tree(ft: dict | None) -> bool {
    """
    Validates file tree structure has required fields.
    
    Args:
        ft: File tree object to validate
        
    Returns:
        bool: True if valid structure, False otherwise
    """
    
    if not ft {
        return False
    }
    
    # Check for required fields
    if "files" not in ft {
        return False
    }
    
    # Files should be a list
    if ft["files"] is not None and not isinstance(ft["files"], list) {
        return False
    }
    
    return True
}

# ============================================================================
# Metadata Generation
# ============================================================================

can _generate_metadata(clone_result, file_tree: dict | None, readme: str) -> dict {
    """
    Generates metadata about the cloned repository.
    
    Args:
        clone_result: Result from clone operation
        file_tree: Built file tree
        readme: README summary
        
    Returns:
        dict: Metadata information
    """
    
    metadata: dict = {
        "clone_timestamp": _get_timestamp(),
        "file_count": 0,
        "directory_count": 0,
        "has_readme": False,
        "readme_length": 0,
        "languages_detected": [],
        "repo_name": clone_result.repo_name if hasattr(clone_result, 'repo_name') else "unknown",
        "size_category": "unknown"
    }
    
    # Count files and directories
    if file_tree and "files" in file_tree and file_tree["files"] {
        metadata["file_count"] = len(file_tree["files"])
        
        # Categorize repository size
        file_count = metadata["file_count"]
        if file_count < 10 {
            metadata["size_category"] = "tiny"
        } elif file_count < 50 {
            metadata["size_category"] = "small"
        } elif file_count < 200 {
            metadata["size_category"] = "medium"
        } elif file_count < 1000 {
            metadata["size_category"] = "large"
        } else {
            metadata["size_category"] = "very_large"
        }
    }
    
    if file_tree and "directories" in file_tree and file_tree["directories"] {
        metadata["directory_count"] = len(file_tree["directories"])
    }
    
    # Check for README
    if readme and len(readme) > 0 and readme != "No README summary available" {
        metadata["has_readme"] = True
        metadata["readme_length"] = len(readme)
    }
    
    # Detect languages from file extensions
    if file_tree and "files" in file_tree and file_tree["files"] {
        metadata["languages_detected"] = _detect_languages(file_tree["files"])
    }
    
    return metadata
}

can _detect_languages(files: list) -> list {
    """
    Detects programming languages from file extensions.
    
    Args:
        files: List of file paths
        
    Returns:
        list: Unique list of detected languages, sorted by frequency
    """
    
    extension_map: dict = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".java": "Java",
        ".go": "Go",
        ".rs": "Rust",
        ".cpp": "C++",
        ".cc": "C++",
        ".cxx": "C++",
        ".c": "C",
        ".h": "C/C++",
        ".hpp": "C++",
        ".jac": "Jac",
        ".rb": "Ruby",
        ".php": "PHP",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".sh": "Shell",
        ".bash": "Bash",
        ".sql": "SQL",
        ".r": "R",
        ".m": "MATLAB",
        ".lua": "Lua",
        ".pl": "Perl",
        ".dart": "Dart",
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".hs": "Haskell",
        ".ml": "OCaml",
        ".vim": "Vimscript",
        ".css": "CSS",
        ".scss": "SCSS",
        ".html": "HTML",
        ".xml": "XML",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".md": "Markdown"
    }
    
    language_count: dict = {}
    
    for file in files {
        for ext in extension_map {
            if file.endswith(ext) {
                lang = extension_map[ext]
                if lang in language_count {
                    language_count[lang] = language_count[lang] + 1
                } else {
                    language_count[lang] = 1
                }
                break
            }
        }
    }
    
    # Sort languages by frequency (most common first)
    sorted_langs = sorted(
        language_count.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return [lang for lang, count in sorted_langs]
}

# ============================================================================
# Status Determination
# ============================================================================

can _determine_clone_status(clone_result) -> str {
    """
    Determines appropriate status from clone result.
    
    Args:
        clone_result: Result from clone operation
        
    Returns:
        str: CloneStatus enum value
    """
    
    if not clone_result or not hasattr(clone_result, 'msg') or not clone_result.msg {
        return CloneStatus.ERROR
    }
    
    msg_lower = clone_result.msg.lower()
    
    if "timeout" in msg_lower or "timed out" in msg_lower {
        return CloneStatus.TIMEOUT
    }
    
    if "too large" in msg_lower or "size limit" in msg_lower or "exceeds" in msg_lower {
        return CloneStatus.TOO_LARGE
    }
    
    if "invalid" in msg_lower or "not found" in msg_lower or "404" in msg_lower {
        return CloneStatus.INVALID_URL
    }
    
    return CloneStatus.ERROR
}

# ============================================================================
# Utility Functions
# ============================================================================

can _get_timestamp() -> str {
    """Returns current timestamp in ISO 8601 format."""
    import:py datetime;
    return datetime.datetime.now().isoformat() + "Z"
}

can _format_file_size(size_bytes: int) -> str {
    """
    Formats file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size (e.g., "1.5 MB")
    """
    
    if size_bytes < 1024 {
        return str(size_bytes) + " B"
    } elif size_bytes < 1024 * 1024 {
        return str(round(size_bytes / 1024, 2)) + " KB"
    } elif size_bytes < 1024 * 1024 * 1024 {
        return str(round(size_bytes / (1024 * 1024), 2)) + " MB"
    } else {
        return str(round(size_bytes / (1024 * 1024 * 1024), 2)) + " GB"
    }
}

# ============================================================================
# Cleanup Walker
# ============================================================================

walker cleanup_repo(path: str) -> dict {
    """
    Cleans up cloned repository directory.
    
    Args:
        path: Repository path to clean up
        
    Returns:
        dict: Result of cleanup operation
            {
                "success": bool,
                "msg": str,
                "timestamp": str
            }
    """
    
    result: dict = {
        "success": False,
        "msg": "",
        "timestamp": _get_timestamp()
    }
    
    if not path or len(path) == 0 {
        result["msg"] = "Invalid path for cleanup"
        return result
    }
    
    # Security check: Don't allow cleaning up root or home directories
    dangerous_paths = ["/", "/home", "/root", "/usr", "/etc", "/var"]
    if path in dangerous_paths or path.startswith("/home/") and path.count("/") <= 2 {
        result["msg"] = "Refusing to clean up potentially dangerous path"
        return result
    }
    
    try {
        cleanup_result = cleanup_directory(path)
        
        if cleanup_result and hasattr(cleanup_result, 'success') and cleanup_result.success {
            result["success"] = True
            result["msg"] = "Repository cleaned up successfully"
        } else {
            error_msg = cleanup_result.msg if hasattr(cleanup_result, 'msg') else "Unknown error"
            result["msg"] = "Cleanup failed: " + error_msg
        }
        
    } except Exception as e {
        result["msg"] = "Cleanup threw exception: " + str(e)
    }
    
    return result
}

# ============================================================================
# Quick Validation Walker
# ============================================================================

walker validate_before_clone(url: str) -> dict {
    """
    Lightweight validation before attempting to clone.
    
    Args:
        url: Repository URL to validate
        
    Returns:
        dict: Validation result
            {
                "valid": bool,
                "msg": str,
                "source": str | None,
                "timestamp": str
            }
    """
    
    validation = _validate_repo_url(url)
    
    return {
        "valid": validation["valid"],
        "msg": validation["msg"],
        "source": validation["source"],
        "timestamp": _get_timestamp()
    }
}

# ============================================================================
# Batch Clone Walker
# ============================================================================

walker clone_batch(urls: list[str]) -> dict {
    """
    Clones multiple repositories in batch.
    
    Args:
        urls: List of repository URLs to clone
        
    Returns:
        dict: Batch results with individual results and summary
            {
                "success": bool,
                "msg": str,
                "results": list[dict],
                "summary": {
                    "total": int,
                    "successful": int,
                    "failed": int,
                    "skipped": int
                },
                "timestamp": str
            }
    """
    
    if not urls or len(urls) == 0 {
        return {
            "success": False,
            "msg": "No URLs provided",
            "results": [],
            "summary": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0
            },
            "timestamp": _get_timestamp()
        }
    }
    
    results: list = []
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    
    for url in urls {
        # Skip empty or invalid entries
        if not url or len(url.strip()) == 0 {
            skipped = skipped + 1
            continue
        }
        
        try {
            result = clone_and_map(url)
            results.append({
                "url": url,
                "result": result
            })
            
            if result["success"] {
                successful = successful + 1
            } else {
                failed = failed + 1
            }
        } except Exception as e {
            results.append({
                "url": url,
                "result": {
                    "success": False,
                    "status": CloneStatus.ERROR,
                    "msg": "Exception during batch processing: " + str(e),
                    "data": None,
                    "errors": [{"step": "batch", "error": str(e)}],
                    "timestamp": _get_timestamp()
                }
            })
            failed = failed + 1
        }
    }
    
    return {
        "success": successful > 0,
        "msg": f"Batch processing completed: {successful} successful, {failed} failed, {skipped} skipped",
        "results": results,
        "summary": {
            "total": len(urls),
            "successful": successful,
            "failed": failed,
            "skipped": skipped
        },
        "timestamp": _get_timestamp()
    }
}

# ============================================================================
# Repository Info Walker (without cloning)
# ============================================================================

walker get_repo_info(url: str) -> dict {
    """
    Gets basic repository information without cloning.
    
    This is a lightweight operation that extracts info from the URL.
    
    Args:
        url: Repository URL
        
    Returns:
        dict: Repository info extracted from URL
            {
                "success": bool,
                "msg": str,
                "data": {
                    "source": str,
                    "owner": str,
                    "repo_name": str,
                    "url": str
                } | None,
                "timestamp": str
            }
    """
    
    result: dict = {
        "success": False,
        "msg": "",
        "data": None,
        "timestamp": _get_timestamp()
    }
    
    validation = _validate_repo_url(url)
    if not validation["valid"] {
        result["msg"] = validation["msg"]
        return result
    }
    
    # Extract owner and repo name from URL
    # Format: https://github.com/owner/repo or https://github.com/owner/repo.git
    try {
        url_clean = url.rstrip("/").replace(".git", "")
        parts = url_clean.split("/")
        
        if len(parts) >= 5 {
            owner = parts[-2]
            repo_name = parts[-1]
            
            result["success"] = True
            result["msg"] = "Repository info extracted successfully"
            result["data"] = {
                "source": validation["source"],
                "owner": owner,
                "repo_name": repo_name,
                "url": url
            }
        } else {
            result["msg"] = "Unable to parse repository owner and name from URL"
        }
    } except Exception as e {
        result["msg"] = "Error parsing URL: " + str(e)
    }
    
    return result
}