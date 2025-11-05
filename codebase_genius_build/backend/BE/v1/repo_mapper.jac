# repo_mapper.jac - Enhanced repository mapping with robust error handling
import py_module;

# Configuration for repository operations
glob REPO_CONFIG = {
    "max_repo_size_mb": 500,
    "clone_timeout_seconds": 300,
    "allowed_extensions": [".py", ".js", ".java", ".go", ".rs", ".cpp", ".c", ".jac"],
    "ignored_dirs": ["node_modules", ".git", "__pycache__", "venv", "env", "dist", "build"]
};

# Result status enum for consistency
enum CloneStatus {
    SUCCESS = "success",
    ERROR = "error",
    TIMEOUT = "timeout",
    TOO_LARGE = "too_large"
}

walker clone_and_map(url: string) -> dict {
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
                "msg": string,
                "data": {
                    "repo_name": string,
                    "path": string,
                    "file_tree": dict,
                    "readme_summary": string,
                    "metadata": dict
                } | null,
                "errors": list,
                "timestamp": string
            }
    
    Example:
        result = clone_and_map("https://github.com/user/repo");
        if result["success"] {
            process_files(result["data"]["file_tree"]);
        }
    """
    
    emit("repo_mapper:start", {"url": url});
    
    # Initialize result structure
    result = {
        "success": false,
        "status": CloneStatus.ERROR,
        "msg": "",
        "data": null,
        "errors": [],
        "timestamp": _get_timestamp()
    };
    
    # Step 1: Validate URL
    emit("repo_mapper:step", {"step": "validation", "status": "started"});
    
    validation = _validate_repo_url(url);
    if !validation["valid"] {
        result["msg"] = "URL validation failed: " + validation["msg"];
        result["errors"].append({
            "step": "validation",
            "error": validation["msg"]
        });
        emit("repo_mapper:error", result);
        ret result;
    }
    
    emit("repo_mapper:step", {"step": "validation", "status": "completed"});
    
    # Step 2: Clone repository
    emit("repo_mapper:step", {"step": "cloning", "status": "started"});
    
    try {
        r = py_module.clone(url);
        
        if !r {
            result["msg"] = "Clone operation returned null";
            result["errors"].append({
                "step": "cloning",
                "error": "Null response from clone operation"
            });
            emit("repo_mapper:error", result);
            ret result;
        }
        
        if !r.success {
            result["msg"] = "Repository cloning failed: " + (r.msg if r.msg else "Unknown error");
            result["status"] = _determine_clone_status(r);
            result["errors"].append({
                "step": "cloning",
                "error": r.msg if r.msg else "Unknown clone error"
            });
            emit("repo_mapper:error", result);
            ret result;
        }
        
        # Validate clone result has required fields
        if !r.path or !r.repo_name {
            result["msg"] = "Clone succeeded but missing required data (path or repo_name)";
            result["errors"].append({
                "step": "cloning",
                "error": "Missing required fields in clone result"
            });
            emit("repo_mapper:error", result);
            ret result;
        }
        
        emit("repo_mapper:step", {
            "step": "cloning",
            "status": "completed",
            "repo_name": r.repo_name,
            "path": r.path
        });
        
    } catch e {
        result["msg"] = "Clone operation threw exception: " + str(e);
        result["errors"].append({
            "step": "cloning",
            "error": str(e)
        });
        emit("repo_mapper:error", result);
        ret result;
    }
    
    # Step 3: Build file tree
    emit("repo_mapper:step", {"step": "file_tree", "status": "started"});
    
    ft = null;
    try {
        ft = py_module.filetree_build(r.path);
        
        if !ft {
            result["errors"].append({
                "step": "file_tree",
                "error": "File tree generation returned null"
            });
            emit("repo_mapper:warning", {
                "msg": "File tree generation failed, using empty tree"
            });
            ft = {"files": [], "directories": []};
        }
        
        # Validate file tree structure
        if !_validate_file_tree(ft) {
            result["errors"].append({
                "step": "file_tree",
                "error": "Invalid file tree structure"
            });
            ft = {"files": [], "directories": []};
        }
        
        emit("repo_mapper:step", {
            "step": "file_tree",
            "status": "completed",
            "file_count": len(ft.files) if ft and ft.files else 0
        });
        
    } catch e {
        result["errors"].append({
            "step": "file_tree",
            "error": str(e)
        });
        emit("repo_mapper:warning", {
            "msg": "File tree building failed: " + str(e)
        });
        ft = {"files": [], "directories": []};
    }
    
    # Step 4: Extract and summarize README
    emit("repo_mapper:step", {"step": "readme", "status": "started"});
    
    rs = null;
    try {
        rs = py_module.readme_summary(r.path);
        
        if !rs {
            result["errors"].append({
                "step": "readme",
                "error": "README summary returned null"
            });
            emit("repo_mapper:warning", {
                "msg": "README summary generation failed"
            });
            rs = "No README summary available";
        }
        
        emit("repo_mapper:step", {
            "step": "readme",
            "status": "completed",
            "summary_length": len(rs) if rs else 0
        });
        
    } catch e {
        result["errors"].append({
            "step": "readme",
            "error": str(e)
        });
        emit("repo_mapper:warning", {
            "msg": "README summarization failed: " + str(e)
        });
        rs = "No README summary available";
    }
    
    # Step 5: Generate metadata
    metadata = _generate_metadata(r, ft, rs);
    
    # Build successful result
    result["success"] = true;
    result["status"] = CloneStatus.SUCCESS;
    result["msg"] = "Repository mapped successfully" + (
        " with warnings" if len(result["errors"]) > 0 else ""
    );
    result["data"] = {
        "repo_name": r.repo_name,
        "path": r.path,
        "file_tree": ft,
        "readme_summary": rs,
        "metadata": metadata
    };
    
    emit("repo_mapper:complete", {
        "repo_name": r.repo_name,
        "file_count": len(ft.files) if ft and ft.files else 0,
        "has_warnings": len(result["errors"]) > 0
    });
    
    ret result;
}

# Validation helper
can _validate_repo_url(url: string) -> dict {
    """
    Validates repository URL format and source.
    
    Args:
        url: Repository URL to validate
        
    Returns:
        dict: {"valid": bool, "msg": string, "source": string|null}
    """
    
    result = {
        "valid": false,
        "msg": "",
        "source": null
    };
    
    # Check for empty URL
    if !url or len(url) == 0 {
        result["msg"] = "URL cannot be empty";
        ret result;
    }
    
    # Check for basic URL structure
    if !("http://" in url or "https://" in url) {
        result["msg"] = "URL must start with http:// or https://";
        ret result;
    }
    
    # Check for supported hosting services
    supported_sources = {
        "github.com": "GitHub",
        "gitlab.com": "GitLab",
        "bitbucket.org": "Bitbucket"
    };
    
    source_found = false;
    for domain in supported_sources {
        if domain in url {
            source_found = true;
            result["source"] = supported_sources[domain];
            break;
        }
    }
    
    if !source_found {
        result["msg"] = "URL must be from a supported source (GitHub, GitLab, or Bitbucket)";
        ret result;
    }
    
    # Check for .git suffix (optional but common)
    # This is informational, not an error
    
    result["valid"] = true;
    result["msg"] = "Valid " + result["source"] + " repository URL";
    ret result;
}

# File tree validation helper
can _validate_file_tree(ft: dict) -> bool {
    """
    Validates file tree structure has required fields.
    
    Args:
        ft: File tree object to validate
        
    Returns:
        bool: True if valid structure, False otherwise
    """
    
    if !ft {
        ret false;
    }
    
    # Check for required fields
    if !("files" in ft) {
        ret false;
    }
    
    # Files should be a list
    if ft.files and !(type(ft.files) == "list") {
        ret false;
    }
    
    ret true;
}

# Metadata generation helper
can _generate_metadata(clone_result: dict, file_tree: dict, readme: string) -> dict {
    """
    Generates metadata about the cloned repository.
    
    Args:
        clone_result: Result from clone operation
        file_tree: Built file tree
        readme: README summary
        
    Returns:
        dict: Metadata information
    """
    
    metadata = {
        "clone_timestamp": _get_timestamp(),
        "file_count": 0,
        "directory_count": 0,
        "has_readme": false,
        "languages_detected": []
    };
    
    # Count files and directories
    if file_tree and file_tree.files {
        metadata["file_count"] = len(file_tree.files);
    }
    
    if file_tree and file_tree.directories {
        metadata["directory_count"] = len(file_tree.directories);
    }
    
    # Check for README
    if readme and len(readme) > 0 and readme != "No README summary available" {
        metadata["has_readme"] = true;
    }
    
    # Detect languages from file extensions (simplified)
    if file_tree and file_tree.files {
        metadata["languages_detected"] = _detect_languages(file_tree.files);
    }
    
    ret metadata;
}

# Language detection helper
can _detect_languages(files: list) -> list {
    """
    Detects programming languages from file extensions.
    
    Args:
        files: List of file paths
        
    Returns:
        list: Unique list of detected languages
    """
    
    extension_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".java": "Java",
        ".go": "Go",
        ".rs": "Rust",
        ".cpp": "C++",
        ".c": "C",
        ".jac": "Jac",
        ".rb": "Ruby",
        ".php": "PHP"
    };
    
    languages = [];
    
    for file in files {
        for ext in extension_map {
            if file.endswith(ext) {
                lang = extension_map[ext];
                if !(lang in languages) {
                    languages.append(lang);
                }
            }
        }
    }
    
    ret languages;
}

# Status determination helper
can _determine_clone_status(clone_result: dict) -> string {
    """
    Determines appropriate status from clone result.
    
    Args:
        clone_result: Result from clone operation
        
    Returns:
        string: CloneStatus enum value
    """
    
    if !clone_result or !clone_result.msg {
        ret CloneStatus.ERROR;
    }
    
    msg_lower = clone_result.msg.lower();
    
    if "timeout" in msg_lower {
        ret CloneStatus.TIMEOUT;
    }
    
    if "too large" in msg_lower or "size limit" in msg_lower {
        ret CloneStatus.TOO_LARGE;
    }
    
    ret CloneStatus.ERROR;
}

# Timestamp helper
can _get_timestamp() -> string {
    """Returns current timestamp in ISO 8601 format."""
    # This would use actual system time in production
    ret "2025-11-03T00:00:00Z";
}

# Cleanup walker for removing cloned repositories
walker cleanup_repo(path: string) -> dict {
    """
    Cleans up cloned repository directory.
    
    Args:
        path: Repository path to clean up
        
    Returns:
        dict: Result of cleanup operation
    """
    
    emit("repo_mapper:cleanup", {"path": path, "status": "started"});
    
    result = {
        "success": false,
        "msg": "",
        "timestamp": _get_timestamp()
    };
    
    if !path or len(path) == 0 {
        result["msg"] = "Invalid path for cleanup";
        ret result;
    }
    
    try {
        # Call Python module cleanup function
        cleanup_result = py_module.cleanup_directory(path);
        
        if cleanup_result and cleanup_result.success {
            result["success"] = true;
            result["msg"] = "Repository cleaned up successfully";
            emit("repo_mapper:cleanup", {"path": path, "status": "completed"});
        } else {
            result["msg"] = "Cleanup failed: " + (cleanup_result.msg if cleanup_result else "Unknown error");
            emit("repo_mapper:cleanup", {"path": path, "status": "failed"});
        }
        
    } catch e {
        result["msg"] = "Cleanup threw exception: " + str(e);
        emit("repo_mapper:cleanup_error", {"path": path, "error": str(e)});
    }
    
    ret result;
}

# Quick validation walker (can be used before cloning)
walker validate_before_clone(url: string) -> dict {
    """
    Lightweight validation before attempting to clone.
    
    Args:
        url: Repository URL to validate
        
    Returns:
        dict: Validation result
    """
    
    validation = _validate_repo_url(url);
    
    ret {
        "valid": validation["valid"],
        "msg": validation["msg"],
        "source": validation["source"],
        "timestamp": _get_timestamp()
    };
}