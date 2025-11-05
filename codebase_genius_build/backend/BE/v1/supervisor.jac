# supervisor.jac - Enhanced orchestrator with robust error handling
import repo_mapper;
import code_analyzer;
import docgenie;

# Configuration constants
glob CONFIG = {
    "max_file_size": 10000000,  # 10MB
    "timeout_seconds": 300,
    "retry_attempts": 3
};

# Result type for standardized responses
enum ResultStatus {
    SUCCESS = "success",
    ERROR = "error",
    PARTIAL = "partial"
}

walker start(url: string) -> dict {
    """
    Main orchestrator walker for documentation generation pipeline.
    
    Args:
        url: Repository URL to process
        
    Returns:
        dict with structure:
        {
            "status": ResultStatus,
            "success": bool,
            "data": dict | null,
            "msg": string,
            "errors": list,
            "timestamp": string
        }
    """
    
    emit("supervisor:start", {"url": url, "timestamp": _get_timestamp()});
    
    # Initialize result structure
    result = {
        "status": ResultStatus.ERROR,
        "success": false,
        "data": null,
        "msg": "",
        "errors": [],
        "timestamp": _get_timestamp()
    };
    
    # Step 1: Clone and map repository
    emit("supervisor:step", {"step": "repo_mapping", "status": "started"});
    m = repo_mapper::clone_and_map(url);
    
    if !m.success {
        result["msg"] = "Repository mapping failed: " + m.msg;
        result["errors"].append({"step": "repo_mapping", "error": m.msg});
        emit("supervisor:error", result);
        ret result;
    }
    
    emit("supervisor:step", {
        "step": "repo_mapping", 
        "status": "completed",
        "files_found": len(m.data.file_tree.files)
    });
    
    # Step 2: Analyze files
    emit("supervisor:step", {"step": "code_analysis", "status": "started"});
    
    try {
        analyses = code_analyzer::analyze_files(
            m.data.path, 
            m.data.file_tree.files
        );
        
        if !analyses or len(analyses) == 0 {
            result["msg"] = "Code analysis produced no results";
            result["errors"].append({
                "step": "code_analysis", 
                "error": "No analyses generated"
            });
            emit("supervisor:warning", result);
            # Continue with empty analyses rather than failing
            analyses = [];
        }
        
        emit("supervisor:step", {
            "step": "code_analysis",
            "status": "completed",
            "analyses_count": len(analyses)
        });
    } catch e {
        result["msg"] = "Code analysis failed: " + str(e);
        result["errors"].append({"step": "code_analysis", "error": str(e)});
        emit("supervisor:error", result);
        ret result;
    }
    
    # Step 3: Build code call graph
    emit("supervisor:step", {"step": "ccg_building", "status": "started"});
    
    try {
        ccg = code_analyzer::build_ccg(m.data.path, analyses);
        
        if !ccg {
            result["errors"].append({
                "step": "ccg_building",
                "error": "CCG generation returned null"
            });
            ccg = null;  # Continue without CCG
        }
        
        emit("supervisor:step", {
            "step": "ccg_building",
            "status": "completed",
            "has_ccg": ccg != null
        });
    } catch e {
        result["errors"].append({
            "step": "ccg_building",
            "error": str(e)
        });
        ccg = null;  # Continue without CCG
        emit("supervisor:warning", {
            "msg": "CCG building failed, continuing without it",
            "error": str(e)
        });
    }
    
    # Step 4: Generate documentation
    emit("supervisor:step", {"step": "doc_generation", "status": "started"});
    
    try {
        out = docgenie::generate(
            m.data.path,
            m.data.repo_name,
            m.data.readme_summary,
            analyses,
            ccg  # Now passing CCG to docgenie
        );
        
        if !out or !out.success {
            result["msg"] = "Documentation generation failed: " + (out.msg if out else "Unknown error");
            result["errors"].append({
                "step": "doc_generation",
                "error": out.msg if out else "Unknown error"
            });
            emit("supervisor:error", result);
            ret result;
        }
        
        emit("supervisor:step", {
            "step": "doc_generation",
            "status": "completed"
        });
        
        # Success path
        result["status"] = ResultStatus.SUCCESS if len(result["errors"]) == 0 else ResultStatus.PARTIAL;
        result["success"] = true;
        result["data"] = out.data;
        result["msg"] = "Documentation generated successfully" + (
            " with warnings" if len(result["errors"]) > 0 else ""
        );
        
        emit("supervisor:complete", result);
        ret result;
        
    } catch e {
        result["msg"] = "Documentation generation failed: " + str(e);
        result["errors"].append({
            "step": "doc_generation",
            "error": str(e)
        });
        emit("supervisor:error", result);
        ret result;
    }
}

# Helper walker for cleanup
walker cleanup(path: string) -> bool {
    """
    Cleanup walker to remove temporary files and directories.
    
    Args:
        path: Path to cleanup
        
    Returns:
        bool indicating success
    """
    try {
        # Call cleanup utilities
        emit("supervisor:cleanup", {"path": path});
        # Add actual cleanup logic here
        ret true;
    } catch e {
        emit("supervisor:cleanup_error", {"path": path, "error": str(e)});
        ret false;
    }
}

# Helper function to get timestamp
can _get_timestamp() -> string {
    # Implementation would use system time
    # Placeholder return
    ret "2025-11-03T00:00:00Z";
}

# Validation walker
walker validate_url(url: string) -> dict {
    """
    Validates repository URL before processing.
    
    Args:
        url: Repository URL to validate
        
    Returns:
        dict with validation results
    """
    result = {
        "valid": false,
        "msg": "",
        "url_type": null
    };
    
    # Basic URL validation
    if !url or len(url) == 0 {
        result["msg"] = "URL cannot be empty";
        ret result;
    }
    
    # Check for common repo hosting services
    valid_domains = ["github.com", "gitlab.com", "bitbucket.org"];
    is_valid_domain = false;
    
    for domain in valid_domains {
        if domain in url {
            is_valid_domain = true;
            result["url_type"] = domain;
            break;
        }
    }
    
    if !is_valid_domain {
        result["msg"] = "URL must be from a supported repository hosting service";
        ret result;
    }
    
    result["valid"] = true;
    result["msg"] = "URL is valid";
    ret result;
}

# Main entry point with validation
walker process_repo(url: string) -> dict {
    """
    Main entry point with URL validation and error recovery.
    
    Args:
        url: Repository URL to process
        
    Returns:
        dict with processing results
    """
    
    # Validate URL first
    validation = validate_url(url);
    if !validation["valid"] {
        ret {
            "status": ResultStatus.ERROR,
            "success": false,
            "msg": "Validation failed: " + validation["msg"],
            "errors": [{"step": "validation", "error": validation["msg"]}],
            "timestamp": _get_timestamp()
        };
    }
    
    # Process with main walker
    result = start(url);
    
    # Cleanup if needed (always cleanup temp files)
    # cleanup would be called here with appropriate paths
    
    ret result;
}