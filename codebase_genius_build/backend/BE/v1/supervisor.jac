# supervisor.jac - Enhanced orchestrator with robust error handling
import:jac from repo_mapper { clone_and_map, cleanup_repo, validate_before_clone }
import:jac from code_analyzer { analyze_files, build_ccg }
import:jac from docgenie { generate }

# Configuration constants
glob CONFIG: dict = {
    "max_file_size": 10000000,  # 10MB
    "timeout_seconds": 300,
    "retry_attempts": 3,
    "auto_cleanup": True,
    "generate_all_formats": True
}

# Result type for standardized responses
enum ResultStatus {
    SUCCESS = "success",
    ERROR = "error",
    PARTIAL = "partial",
    VALIDATION_ERROR = "validation_error"
}

# Pipeline stage tracking
enum PipelineStage {
    VALIDATION = "validation",
    REPO_MAPPING = "repo_mapping",
    CODE_ANALYSIS = "code_analysis",
    CCG_BUILDING = "ccg_building",
    DOC_GENERATION = "doc_generation",
    CLEANUP = "cleanup"
}

# ============================================================================
# Main Orchestrator Walker
# ============================================================================

walker start(url: str, options: dict | None = None) -> dict {
    """
    Main orchestrator walker for documentation generation pipeline.
    
    This walker orchestrates the complete documentation generation process:
    1. Clone and map repository
    2. Analyze code files
    3. Build code call graph
    4. Generate documentation
    5. Cleanup (optional)
    
    Args:
        url: Repository URL to process
        options: Optional configuration overrides
        
    Returns:
        dict with structure:
        {
            "status": ResultStatus,
            "success": bool,
            "data": dict | None,
            "msg": str,
            "errors": list,
            "warnings": list,
            "stages_completed": list,
            "timestamp": str,
            "duration_ms": int
        }
    
    Example:
        result = start("https://github.com/user/repo");
        if result["success"] {
            print("Docs at: " + result["data"]["output_path"]);
        }
    """
    
    start_time = _get_timestamp_ms()
    
    # Initialize result structure
    result: dict = {
        "status": ResultStatus.ERROR,
        "success": False,
        "data": None,
        "msg": "",
        "errors": [],
        "warnings": [],
        "stages_completed": [],
        "timestamp": _get_timestamp(),
        "duration_ms": 0
    }
    
    # Merge options with defaults
    opts = _merge_options(options)
    
    # Step 1: Clone and map repository
    result["current_stage"] = PipelineStage.REPO_MAPPING
    
    try {
        m = clone_and_map(url)
        
        if not m["success"] {
            result["msg"] = "Repository mapping failed: " + m["msg"]
            result["errors"].append({
                "stage": PipelineStage.REPO_MAPPING,
                "error": m["msg"]
            })
            result["duration_ms"] = _get_timestamp_ms() - start_time
            return result
        }
        
        result["stages_completed"].append(PipelineStage.REPO_MAPPING)
        
        # Validate mapped data
        if not m["data"] or not m["data"]["file_tree"] or not m["data"]["file_tree"]["files"] {
            result["msg"] = "Repository mapped but no files found"
            result["errors"].append({
                "stage": PipelineStage.REPO_MAPPING,
                "error": "No files found in repository"
            })
            result["duration_ms"] = _get_timestamp_ms() - start_time
            return result
        }
        
        # Store mapping errors as warnings
        if "errors" in m and len(m["errors"]) > 0 {
            result["warnings"].extend(m["errors"])
        }
        
    } except Exception as e {
        result["msg"] = "Repository mapping threw exception: " + str(e)
        result["errors"].append({
            "stage": PipelineStage.REPO_MAPPING,
            "error": str(e)
        })
        result["duration_ms"] = _get_timestamp_ms() - start_time
        return result
    }
    
    # Step 2: Analyze files
    result["current_stage"] = PipelineStage.CODE_ANALYSIS
    
    analyses: list = []
    try {
        analyses = analyze_files(
            m["data"]["path"],
            m["data"]["file_tree"]["files"]
        )
        
        if not analyses or len(analyses) == 0 {
            result["warnings"].append({
                "stage": PipelineStage.CODE_ANALYSIS,
                "warning": "No analyses generated, continuing with empty list"
            })
            analyses = []
        }
        
        result["stages_completed"].append(PipelineStage.CODE_ANALYSIS)
        
    } except Exception as e {
        result["warnings"].append({
            "stage": PipelineStage.CODE_ANALYSIS,
            "warning": "Code analysis failed: " + str(e)
        })
        analyses = []  # Continue with empty analyses
    }
    
    # Step 3: Build code call graph
    result["current_stage"] = PipelineStage.CCG_BUILDING
    
    ccg: dict | None = None
    try {
        ccg = build_ccg(m["data"]["path"], analyses)
        
        if not ccg {
            result["warnings"].append({
                "stage": PipelineStage.CCG_BUILDING,
                "warning": "CCG generation returned null, continuing without it"
            })
        }
        
        result["stages_completed"].append(PipelineStage.CCG_BUILDING)
        
    } except Exception as e {
        result["warnings"].append({
            "stage": PipelineStage.CCG_BUILDING,
            "warning": "CCG building failed: " + str(e)
        })
        ccg = None  # Continue without CCG
    }
    
    # Step 4: Generate documentation
    result["current_stage"] = PipelineStage.DOC_GENERATION
    
    try {
        # Prepare generation options
        gen_options = {
            "generate_html": opts["generate_all_formats"],
            "generate_ccg": ccg is not None,
            "generate_api_docs": True,
            "generate_architecture": True
        }
        
        out = generate(
            m["data"]["path"],
            m["data"]["repo_name"],
            m["data"]["readme_summary"],
            analyses,
            ccg,
            gen_options
        )
        
        if not out or not out["success"] {
            error_msg = out["msg"] if out and "msg" in out else "Unknown error"
            result["msg"] = "Documentation generation failed: " + error_msg
            result["errors"].append({
                "stage": PipelineStage.DOC_GENERATION,
                "error": error_msg
            })
            result["duration_ms"] = _get_timestamp_ms() - start_time
            return result
        }
        
        result["stages_completed"].append(PipelineStage.DOC_GENERATION)
        
        # Add generation warnings to result
        if "warnings" in out and len(out["warnings"]) > 0 {
            result["warnings"].extend(out["warnings"])
        }
        
        # Success path
        result["status"] = ResultStatus.SUCCESS if len(result["errors"]) == 0 else ResultStatus.PARTIAL
        result["success"] = True
        result["data"] = out["data"]
        result["msg"] = "Documentation generated successfully" + (
            f" with {len(result['warnings'])} warnings" if len(result["warnings"]) > 0 else ""
        )
        
        # Step 5: Cleanup (if enabled)
        if opts["auto_cleanup"] {
            result["current_stage"] = PipelineStage.CLEANUP
            cleanup_result = _safe_cleanup(m["data"]["path"])
            if cleanup_result["success"] {
                result["stages_completed"].append(PipelineStage.CLEANUP)
            } else {
                result["warnings"].append({
                    "stage": PipelineStage.CLEANUP,
                    "warning": "Cleanup failed: " + cleanup_result["msg"]
                })
            }
        }
        
        result["duration_ms"] = _get_timestamp_ms() - start_time
        return result
        
    } except Exception as e {
        result["msg"] = "Documentation generation threw exception: " + str(e)
        result["errors"].append({
            "stage": PipelineStage.DOC_GENERATION,
            "error": str(e)
        })
        result["duration_ms"] = _get_timestamp_ms() - start_time
        return result
    }
}

# ============================================================================
# Validation Walker
# ============================================================================

walker validate_url(url: str) -> dict {
    """
    Validates repository URL before processing.
    
    Args:
        url: Repository URL to validate
        
    Returns:
        dict with validation results:
        {
            "valid": bool,
            "msg": str,
            "url_type": str | None,
            "source": str | None
        }
    """
    
    result: dict = {
        "valid": False,
        "msg": "",
        "url_type": None,
        "source": None
    }
    
    # Basic URL validation
    if not url or len(url.strip()) == 0 {
        result["msg"] = "URL cannot be empty"
        return result
    }
    
    url = url.strip()
    
    # Check for HTTP/HTTPS protocol
    if not url.startswith("http://") and not url.startswith("https://") {
        result["msg"] = "URL must start with http:// or https://"
        return result
    }
    
    # Check for common repo hosting services
    valid_domains: dict = {
        "github.com": "GitHub",
        "gitlab.com": "GitLab",
        "bitbucket.org": "Bitbucket",
        "codeberg.org": "Codeberg"
    }
    
    is_valid_domain: bool = False
    
    for domain in valid_domains {
        if domain in url {
            is_valid_domain = True
            result["url_type"] = domain
            result["source"] = valid_domains[domain]
            break
        }
    }
    
    if not is_valid_domain {
        result["msg"] = "URL must be from a supported repository hosting service (GitHub, GitLab, Bitbucket, Codeberg)"
        return result
    }
    
    # Check for reasonable URL structure
    if url.count("/") < 3 {
        result["msg"] = "URL appears to be incomplete (missing repository path)"
        return result
    }
    
    result["valid"] = True
    result["msg"] = f"Valid {result['source']} repository URL"
    return result
}

# ============================================================================
# Main Entry Point with Validation
# ============================================================================

walker process_repo(url: str, options: dict | None = None) -> dict {
    """
    Main entry point with URL validation and error recovery.
    
    This is the recommended entry point for external callers as it includes
    pre-validation and comprehensive error handling.
    
    Args:
        url: Repository URL to process
        options: Optional configuration overrides
        
    Returns:
        dict with processing results (same structure as start())
    
    Example:
        result = process_repo(
            "https://github.com/user/repo",
            {"auto_cleanup": False}
        );
    """
    
    # Step 1: Validate URL first
    validation = validate_url(url)
    if not validation["valid"] {
        return {
            "status": ResultStatus.VALIDATION_ERROR,
            "success": False,
            "data": None,
            "msg": "Validation failed: " + validation["msg"],
            "errors": [{
                "stage": PipelineStage.VALIDATION,
                "error": validation["msg"]
            }],
            "warnings": [],
            "stages_completed": [],
            "timestamp": _get_timestamp(),
            "duration_ms": 0
        }
    }
    
    # Step 2: Process with main walker
    result = start(url, options)
    
    return result
}

# ============================================================================
# Batch Processing Walker
# ============================================================================

walker process_batch(urls: list[str], options: dict | None = None) -> dict {
    """
    Process multiple repositories in batch.
    
    Args:
        urls: List of repository URLs to process
        options: Optional configuration overrides applied to all repos
        
    Returns:
        dict with batch results:
        {
            "success": bool,
            "msg": str,
            "results": list[dict],
            "summary": {
                "total": int,
                "successful": int,
                "failed": int,
                "partial": int,
                "skipped": int
            },
            "timestamp": str,
            "total_duration_ms": int
        }
    """
    
    start_time = _get_timestamp_ms()
    
    if not urls or len(urls) == 0 {
        return {
            "success": False,
            "msg": "No URLs provided",
            "results": [],
            "summary": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "partial": 0,
                "skipped": 0
            },
            "timestamp": _get_timestamp(),
            "total_duration_ms": 0
        }
    }
    
    results: list = []
    successful: int = 0
    failed: int = 0
    partial: int = 0
    skipped: int = 0
    
    for url in urls {
        # Skip empty entries
        if not url or len(url.strip()) == 0 {
            skipped = skipped + 1
            continue
        }
        
        try {
            result = process_repo(url, options)
            results.append({
                "url": url,
                "result": result
            })
            
            if result["success"] {
                if result["status"] == ResultStatus.SUCCESS {
                    successful = successful + 1
                } else {
                    partial = partial + 1
                }
            } else {
                failed = failed + 1
            }
        } except Exception as e {
            results.append({
                "url": url,
                "result": {
                    "status": ResultStatus.ERROR,
                    "success": False,
                    "data": None,
                    "msg": "Exception during batch processing: " + str(e),
                    "errors": [{"stage": "batch", "error": str(e)}],
                    "warnings": [],
                    "stages_completed": [],
                    "timestamp": _get_timestamp(),
                    "duration_ms": 0
                }
            })
            failed = failed + 1
        }
    }
    
    total_duration = _get_timestamp_ms() - start_time
    
    return {
        "success": successful > 0 or partial > 0,
        "msg": f"Batch processing completed: {successful} successful, {partial} partial, {failed} failed, {skipped} skipped",
        "results": results,
        "summary": {
            "total": len(urls),
            "successful": successful,
            "failed": failed,
            "partial": partial,
            "skipped": skipped
        },
        "timestamp": _get_timestamp(),
        "total_duration_ms": total_duration
    }
}

# ============================================================================
# Retry Walker
# ============================================================================

walker process_with_retry(url: str, max_retries: int = 3, options: dict | None = None) -> dict {
    """
    Process repository with automatic retry on failure.
    
    Args:
        url: Repository URL to process
        max_retries: Maximum number of retry attempts
        options: Optional configuration overrides
        
    Returns:
        dict with processing results plus retry info:
        {
            ...(standard result fields),
            "retry_count": int,
            "retry_errors": list
        }
    """
    
    retry_count: int = 0
    retry_errors: list = []
    
    while retry_count <= max_retries {
        result = process_repo(url, options)
        
        if result["success"] {
            result["retry_count"] = retry_count
            result["retry_errors"] = retry_errors
            return result
        }
        
        retry_errors.append({
            "attempt": retry_count + 1,
            "errors": result["errors"]
        })
        
        retry_count = retry_count + 1
        
        if retry_count <= max_retries {
            # Wait before retry (exponential backoff could be added here)
            pass
        }
    }
    
    # All retries exhausted
    result = process_repo(url, options)
    result["retry_count"] = retry_count
    result["retry_errors"] = retry_errors
    result["msg"] = result["msg"] + f" (failed after {retry_count} retries)"
    
    return result
}

# ============================================================================
# Helper Functions
# ============================================================================

can _merge_options(options: dict | None) -> dict {
    """
    Merges user options with default configuration.
    
    Args:
        options: User-provided options
        
    Returns:
        dict: Merged options
    """
    
    merged = CONFIG.copy()
    
    if options {
        for key in options {
            merged[key] = options[key]
        }
    }
    
    return merged
}

can _safe_cleanup(path: str) -> dict {
    """
    Safely cleans up repository with error handling.
    
    Args:
        path: Path to clean up
        
    Returns:
        dict: Cleanup result
    """
    
    try {
        return cleanup_repo(path)
    } except Exception as e {
        return {
            "success": False,
            "msg": "Cleanup exception: " + str(e),
            "timestamp": _get_timestamp()
        }
    }
}

can _get_timestamp() -> str {
    """Returns current timestamp in ISO 8601 format."""
    import:py datetime;
    return datetime.datetime.now().isoformat() + "Z"
}

can _get_timestamp_ms() -> int {
    """Returns current timestamp in milliseconds."""
    import:py datetime;
    return int(datetime.datetime.now().timestamp() * 1000)
}

can _format_duration(duration_ms: int) -> str {
    """
    Formats duration in human-readable format.
    
    Args:
        duration_ms: Duration in milliseconds
        
    Returns:
        str: Formatted duration (e.g., "2.5s", "1m 30s")
    """
    
    if duration_ms < 1000 {
        return str(duration_ms) + "ms"
    } elif duration_ms < 60000 {
        seconds = duration_ms / 1000.0
        return f"{seconds:.1f}s"
    } else {
        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) / 1000.0
        return f"{minutes}m {seconds:.1f}s"
    }
}

# ============================================================================
# Status Check Walker
# ============================================================================

walker check_status(result: dict) -> dict {
    """
    Analyzes a pipeline result and provides status summary.
    
    Args:
        result: Result from process_repo or start
        
    Returns:
        dict with status analysis:
        {
            "overall_status": str,
            "stages_summary": dict,
            "error_count": int,
            "warning_count": int,
            "recommendations": list
        }
    """
    
    status_info: dict = {
        "overall_status": result.get("status", "unknown"),
        "stages_summary": {},
        "error_count": len(result.get("errors", [])),
        "warning_count": len(result.get("warnings", [])),
        "recommendations": []
    }
    
    # Analyze stages
    completed_stages = result.get("stages_completed", [])
    all_stages = [
        PipelineStage.REPO_MAPPING,
        PipelineStage.CODE_ANALYSIS,
        PipelineStage.CCG_BUILDING,
        PipelineStage.DOC_GENERATION
    ]
    
    for stage in all_stages {
        status_info["stages_summary"][stage] = "completed" if stage in completed_stages else "not_completed"
    }
    
    # Generate recommendations
    if status_info["error_count"] > 0 {
        status_info["recommendations"].append("Review errors for critical issues")
    }
    
    if status_info["warning_count"] > 5 {
        status_info["recommendations"].append("High number of warnings - consider investigating")
    }
    
    if PipelineStage.CODE_ANALYSIS not in completed_stages {
        status_info["recommendations"].append("Code analysis failed - check file accessibility")
    }
    
    return status_info
}