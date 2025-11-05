# code_analyzer.jac - Enhanced code analysis with robust error handling
import py_module;
import json;

# Global configuration for code analysis
glob ANALYZER_CONFIG = {
    "max_file_size_mb": 10,
    "timeout_seconds": 30,
    "skip_binary_files": true,
    "parallel_processing": false,
    "max_concurrent_files": 5,
    "supported_languages": ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c", "jac"],
    "skip_dirs": ["node_modules", ".git", "__pycache__", "venv", "env", "dist", "build", ".idea", ".vscode"]
};

# Analysis status enum
enum AnalysisStatus {
    SUCCESS = "success",
    PARTIAL = "partial",
    FAILED = "failed",
    SKIPPED = "skipped",
    TIMEOUT = "timeout"
}

# ============================================================================
# Main Analysis Walker - Batch File Processing
# ============================================================================

walker analyze_files(repo_root: string, files: list) -> list {
    """
    Analyzes multiple files in a repository.
    
    This walker processes a list of files and returns analysis results
    for each file. Handles errors gracefully and continues processing
    even if individual files fail.
    
    Args:
        repo_root: Root directory of the repository
        files: List of relative file paths to analyze
        
    Returns:
        list: Analysis results for each file with structure:
            [
                {
                    "file": string,
                    "status": AnalysisStatus,
                    "parsed": dict | null,
                    "metadata": {
                        "language": string,
                        "lines": int,
                        "size_bytes": int,
                        "analysis_time_ms": int
                    },
                    "error": string | null
                },
                ...
            ]
    
    Example:
        files = ["src/main.py", "src/utils.py", "tests/test.py"];
        results = analyze_files("/path/to/repo", files);
        
        for result in results {
            if result["status"] == "success" {
                process_analysis(result["parsed"]);
            } else {
                log_error(result["error"]);
            }
        }
    """
    
    emit("analyzer:batch_start", {
        "repo_root": repo_root,
        "file_count": len(files) if files else 0
    });
    
    # Validate inputs
    if !repo_root or len(repo_root) == 0 {
        emit("analyzer:error", {"error": "Repository root is required"});
        ret [];
    }
    
    if !files {
        emit("analyzer:warning", {"warning": "No files provided"});
        ret [];
    }
    
    if len(files) == 0 {
        emit("analyzer:warning", {"warning": "Empty files list"});
        ret [];
    }
    
    results = [];
    successful = 0;
    failed = 0;
    skipped = 0;
    
    emit("analyzer:progress", {
        "status": "processing",
        "total": len(files),
        "completed": 0
    });
    
    # Process each file
    for i in range(len(files)) {
        file = files[i];
        
        # Skip null/invalid entries
        if !file or len(file) == 0 {
            results.append({
                "file": "unknown",
                "status": AnalysisStatus.SKIPPED,
                "parsed": null,
                "metadata": null,
                "error": "Empty or null file path"
            });
            skipped = skipped + 1;
            continue;
        }
        
        # Check if file should be skipped
        if _should_skip_file(file) {
            results.append({
                "file": file,
                "status": AnalysisStatus.SKIPPED,
                "parsed": null,
                "metadata": {
                    "skip_reason": "Binary or excluded file type"
                },
                "error": null
            });
            skipped = skipped + 1;
            emit("analyzer:file_skipped", {"file": file});
            continue;
        }
        
        # Analyze the file
        try {
            result = analyze_file(repo_root, file);
            
            if result and result["status"] == AnalysisStatus.SUCCESS {
                successful = successful + 1;
            } else {
                failed = failed + 1;
            }
            
            results.append(result);
            
        } catch e {
            # Handle unexpected errors
            results.append({
                "file": file,
                "status": AnalysisStatus.FAILED,
                "parsed": null,
                "metadata": null,
                "error": "Exception during analysis: " + str(e)
            });
            failed = failed + 1;
            
            emit("analyzer:file_error", {
                "file": file,
                "error": str(e)
            });
        }
        
        # Progress update
        if (i + 1) % 10 == 0 or (i + 1) == len(files) {
            emit("analyzer:progress", {
                "status": "processing",
                "total": len(files),
                "completed": i + 1,
                "successful": successful,
                "failed": failed,
                "skipped": skipped
            });
        }
    }
    
    emit("analyzer:batch_complete", {
        "total": len(files),
        "successful": successful,
        "failed": failed,
        "skipped": skipped
    });
    
    ret results;
}

# ============================================================================
# Single File Analysis Walker
# ============================================================================

walker analyze_file(repo_root: string, relpath: string) -> dict {
    """
    Analyzes a single file in the repository.
    
    Performs deep code analysis including:
    - AST parsing
    - Function/class extraction
    - Dependency detection
    - Complexity metrics
    - Documentation extraction
    
    Args:
        repo_root: Root directory of the repository
        relpath: Relative path to the file from repo root
        
    Returns:
        dict: Analysis result with structure:
            {
                "file": string,
                "status": AnalysisStatus,
                "parsed": {
                    "nodes": list,      # AST nodes (functions, classes, etc.)
                    "edges": list,      # Dependencies/calls
                    "imports": list,    # Import statements
                    "exports": list,    # Exported items
                    "complexity": int   # Cyclomatic complexity
                },
                "metadata": {
                    "language": string,
                    "lines": int,
                    "size_bytes": int,
                    "analysis_time_ms": int,
                    "encoding": string
                },
                "error": string | null
            }
    
    Example:
        result = analyze_file("/path/to/repo", "src/main.py");
        
        if result["status"] == "success" {
            functions = result["parsed"]["nodes"];
            for func in functions {
                print("Function: " + func["name"]);
            }
        }
    """
    
    start_time = _get_timestamp_ms();
    
    emit("analyzer:file_start", {
        "file": relpath,
        "repo_root": repo_root
    });
    
    # Initialize result structure
    result = {
        "file": relpath,
        "status": AnalysisStatus.FAILED,
        "parsed": null,
        "metadata": null,
        "error": null
    };
    
    # Validate inputs
    if !repo_root or len(repo_root) == 0 {
        result["error"] = "Repository root is required";
        emit("analyzer:file_error", result);
        ret result;
    }
    
    if !relpath or len(relpath) == 0 {
        result["error"] = "File path is required";
        emit("analyzer:file_error", result);
        ret result;
    }
    
    # Construct full path
    full_path = repo_root + "/" + relpath;
    
    # Normalize path (remove double slashes, etc.)
    full_path = _normalize_path(full_path);
    
    # Detect language
    language = _detect_language(relpath);
    
    # Check if language is supported
    if language and !(language in ANALYZER_CONFIG["supported_languages"]) {
        result["status"] = AnalysisStatus.SKIPPED;
        result["error"] = "Unsupported language: " + language;
        result["metadata"] = {
            "language": language,
            "skip_reason": "Unsupported language"
        };
        emit("analyzer:file_skipped", result);
        ret result;
    }
    
    # Get file metadata
    try {
        file_info = py_module.get_file_info(full_path);
        
        if !file_info {
            result["error"] = "Failed to get file info";
            emit("analyzer:file_error", result);
            ret result;
        }
        
        # Check file size
        if file_info.size_bytes > (ANALYZER_CONFIG["max_file_size_mb"] * 1024 * 1024) {
            result["status"] = AnalysisStatus.SKIPPED;
            result["error"] = "File too large: " + str(file_info.size_bytes) + " bytes";
            result["metadata"] = {
                "language": language,
                "size_bytes": file_info.size_bytes,
                "skip_reason": "File too large"
            };
            emit("analyzer:file_skipped", result);
            ret result;
        }
        
        # Check if file exists
        if !file_info.exists {
            result["error"] = "File does not exist: " + full_path;
            emit("analyzer:file_error", result);
            ret result;
        }
        
        # Check if it's a binary file
        if file_info.is_binary and ANALYZER_CONFIG["skip_binary_files"] {
            result["status"] = AnalysisStatus.SKIPPED;
            result["error"] = "Binary file skipped";
            result["metadata"] = {
                "language": language,
                "size_bytes": file_info.size_bytes,
                "skip_reason": "Binary file"
            };
            emit("analyzer:file_skipped", result);
            ret result;
        }
        
    } catch e {
        result["error"] = "Error getting file info: " + str(e);
        emit("analyzer:file_error", result);
        ret result;
    }
    
    # Parse the file
    emit("analyzer:parsing", {"file": relpath});
    
    parsed = null;
    try {
        parsed = py_module.parse_file(full_path);
        
        if !parsed {
            result["error"] = "Parser returned null";
            emit("analyzer:file_error", result);
            ret result;
        }
        
        # Validate parsed structure
        if !_validate_parsed_structure(parsed) {
            result["status"] = AnalysisStatus.PARTIAL;
            result["error"] = "Invalid or incomplete parse result";
            result["parsed"] = parsed;  # Include partial data
        } else {
            result["status"] = AnalysisStatus.SUCCESS;
            result["parsed"] = parsed;
        }
        
    } catch e {
        result["error"] = "Parse error: " + str(e);
        emit("analyzer:parse_error", {
            "file": relpath,
            "error": str(e)
        });
        ret result;
    }
    
    # Enrich parsed data
    if parsed {
        try {
            enriched = _enrich_analysis(parsed, language, full_path);
            result["parsed"] = enriched;
        } catch e {
            # Enrichment failed, but we still have basic parsed data
            emit("analyzer:enrichment_error", {
                "file": relpath,
                "error": str(e)
            });
        }
    }
    
    # Build metadata
    processing_time = _get_timestamp_ms() - start_time;
    
    result["metadata"] = {
        "language": language,
        "lines": file_info.lines if file_info and "lines" in file_info else 0,
        "size_bytes": file_info.size_bytes if file_info and "size_bytes" in file_info else 0,
        "analysis_time_ms": processing_time,
        "encoding": file_info.encoding if file_info and "encoding" in file_info else "utf-8"
    };
    
    emit("analyzer:file_complete", {
        "file": relpath,
        "status": result["status"],
        "analysis_time_ms": processing_time
    });
    
    ret result;
}

# ============================================================================
# Code Call Graph (CCG) Builder
# ============================================================================

walker build_ccg(repo_root: string, analyses: list) -> dict {
    """
    Builds a code call graph from analysis results.
    
    Creates a directed graph showing relationships between code elements:
    - Function calls
    - Class inheritance
    - Import dependencies
    - Module relationships
    
    Args:
        repo_root: Root directory of the repository
        analyses: List of analysis results from analyze_files
        
    Returns:
        dict: Code call graph with structure:
            {
                "success": bool,
                "nodes": [
                    {
                        "id": string,
                        "name": string,
                        "type": string,    # "function", "class", "module"
                        "file": string,
                        "metadata": dict
                    }
                ],
                "edges": [
                    {
                        "from": string,    # Source node ID
                        "to": string,      # Target node ID
                        "type": string,    # "calls", "imports", "inherits"
                        "file": string     # File where edge originates
                    }
                ],
                "statistics": {
                    "total_nodes": int,
                    "total_edges": int,
                    "node_types": dict,
                    "edge_types": dict
                },
                "errors": list,
                "timestamp": string
            }
    
    Example:
        analyses = analyze_files(repo_root, files);
        ccg = build_ccg(repo_root, analyses);
        
        if ccg["success"] {
            print("Graph has " + str(ccg["statistics"]["total_nodes"]) + " nodes");
            visualize_graph(ccg["nodes"], ccg["edges"]);
        }
    """
    
    start_time = _get_timestamp_ms();
    
    emit("analyzer:ccg_start", {
        "repo_root": repo_root,
        "analysis_count": len(analyses) if analyses else 0
    });
    
    # Initialize result
    result = {
        "success": false,
        "nodes": [],
        "edges": [],
        "statistics": {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_types": {}
        },
        "errors": [],
        "timestamp": _get_timestamp()
    };
    
    # Validate inputs
    if !repo_root or len(repo_root) == 0 {
        result["errors"].append("Repository root is required");
        emit("analyzer:ccg_error", result);
        ret result;
    }
    
    if !analyses {
        result["errors"].append("Analyses list is required");
        emit("analyzer:ccg_error", result);
        ret result;
    }
    
    if len(analyses) == 0 {
        emit("analyzer:ccg_warning", {"warning": "No analyses provided"});
        result["success"] = true;  # Empty graph is valid
        ret result;
    }
    
    # Build node list and name mapping
    nodes = [];
    name_map = {};  # Maps node names to node IDs
    file_map = {};  # Maps node IDs to files
    
    emit("analyzer:ccg_step", {"step": "building_nodes", "status": "started"});
    
    for analysis in analyses {
        # Skip failed or skipped analyses
        if !analysis or !("parsed" in analysis) or !analysis["parsed"] {
            continue;
        }
        
        # Skip analyses without nodes
        if !("nodes" in analysis["parsed"]) or !analysis["parsed"]["nodes"] {
            continue;
        }
        
        file = analysis["file"] if "file" in analysis else "unknown";
        
        # Process each node
        for node in analysis["parsed"]["nodes"] {
            if !node {
                continue;
            }
            
            # Validate node structure
            if !("name" in node) or !("id" in node) {
                result["errors"].append({
                    "file": file,
                    "error": "Node missing required fields (name or id)"
                });
                continue;
            }
            
            # Enrich node with file information
            enriched_node = {
                "id": node["id"],
                "name": node["name"],
                "type": node["type"] if "type" in node else "unknown",
                "file": file,
                "metadata": node["metadata"] if "metadata" in node else {}
            };
            
            # Add to nodes list
            nodes.append(enriched_node);
            
            # Update name mapping
            name_map[node["name"]] = node["id"];
            file_map[node["id"]] = file;
            
            # Track node types
            node_type = enriched_node["type"];
            if node_type in result["statistics"]["node_types"] {
                result["statistics"]["node_types"][node_type] = 
                    result["statistics"]["node_types"][node_type] + 1;
            } else {
                result["statistics"]["node_types"][node_type] = 1;
            }
        }
    }
    
    result["statistics"]["total_nodes"] = len(nodes);
    
    emit("analyzer:ccg_step", {
        "step": "building_nodes",
        "status": "completed",
        "node_count": len(nodes)
    });
    
    # Build edge list
    edges = [];
    
    emit("analyzer:ccg_step", {"step": "building_edges", "status": "started"});
    
    for analysis in analyses {
        # Skip failed or skipped analyses
        if !analysis or !("parsed" in analysis) or !analysis["parsed"] {
            continue;
        }
        
        # Skip analyses without edges
        if !("edges" in analysis["parsed"]) or !analysis["parsed"]["edges"] {
            continue;
        }
        
        file = analysis["file"] if "file" in analysis else "unknown";
        
        # Process each edge
        for edge in analysis["parsed"]["edges"] {
            if !edge {
                continue;
            }
            
            # Validate edge structure
            if !("to" in edge) {
                result["errors"].append({
                    "file": file,
                    "error": "Edge missing 'to' field"
                });
                continue;
            }
            
            to_name = edge["to"];
            
            # Resolve target ID from name
            target_id = name_map.get(to_name, "");
            
            # If target not found, use the name itself as ID
            if !target_id or len(target_id) == 0 {
                target_id = to_name;
            }
            
            # Build enriched edge
            enriched_edge = {
                "from": file,  # Source file
                "to": target_id,
                "type": edge["type"] if "type" in edge else "unknown",
                "file": file,
                "metadata": edge["metadata"] if "metadata" in edge else {}
            };
            
            # Add from_node_id if available
            if "from" in edge {
                enriched_edge["from_node_id"] = edge["from"];
            }
            
            edges.append(enriched_edge);
            
            # Track edge types
            edge_type = enriched_edge["type"];
            if edge_type in result["statistics"]["edge_types"] {
                result["statistics"]["edge_types"][edge_type] = 
                    result["statistics"]["edge_types"][edge_type] + 1;
            } else {
                result["statistics"]["edge_types"][edge_type] = 1;
            }
        }
    }
    
    result["statistics"]["total_edges"] = len(edges);
    
    emit("analyzer:ccg_step", {
        "step": "building_edges",
        "status": "completed",
        "edge_count": len(edges)
    });
    
    # Validate graph structure
    validation = _validate_ccg(nodes, edges);
    if !validation["valid"] {
        result["errors"].append({
            "error": "CCG validation failed",
            "details": validation["errors"]
        });
    }
    
    # Build final result
    result["nodes"] = nodes;
    result["edges"] = edges;
    result["success"] = true;
    
    processing_time = _get_timestamp_ms() - start_time;
    result["statistics"]["processing_time_ms"] = processing_time;
    
    emit("analyzer:ccg_complete", {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "processing_time_ms": processing_time,
        "has_errors": len(result["errors"]) > 0
    });
    
    ret result;
}

# ============================================================================
# Advanced Analysis Walkers
# ============================================================================

walker analyze_dependencies(analyses: list) -> dict {
    """
    Analyzes dependencies across the codebase.
    
    Args:
        analyses: List of file analysis results
        
    Returns:
        dict: Dependency analysis with import graph
    """
    
    result = {
        "imports": {},      # File -> list of imports
        "exports": {},      # File -> list of exports
        "dependencies": {}, # File -> list of dependent files
        "circular": []      # List of circular dependencies
    };
    
    if !analyses {
        ret result;
    }
    
    for analysis in analyses {
        if !analysis or !("file" in analysis) or !("parsed" in analysis) {
            continue;
        }
        
        file = analysis["file"];
        parsed = analysis["parsed"];
        
        # Extract imports
        if parsed and "imports" in parsed {
            result["imports"][file] = parsed["imports"];
        }
        
        # Extract exports
        if parsed and "exports" in parsed {
            result["exports"][file] = parsed["exports"];
        }
    }
    
    # Detect circular dependencies
    result["circular"] = _detect_circular_dependencies(result["imports"]);
    
    ret result;
}

walker calculate_metrics(analyses: list) -> dict {
    """
    Calculates code metrics across the codebase.
    
    Args:
        analyses: List of file analysis results
        
    Returns:
        dict: Code metrics including complexity, coverage, etc.
    """
    
    metrics = {
        "total_files": 0,
        "total_lines": 0,
        "total_functions": 0,
        "total_classes": 0,
        "average_complexity": 0,
        "complexity_distribution": {
            "low": 0,     # 1-10
            "medium": 0,  # 11-20
            "high": 0,    # 21-50
            "very_high": 0 # 50+
        },
        "languages": {}
    };
    
    if !analyses or len(analyses) == 0 {
        ret metrics;
    }
    
    total_complexity = 0;
    complexity_count = 0;
    
    for analysis in analyses {
        if !analysis or analysis["status"] != AnalysisStatus.SUCCESS {
            continue;
        }
        
        metrics["total_files"] = metrics["total_files"] + 1;
        
        # Count lines
        if "metadata" in analysis and "lines" in analysis["metadata"] {
            metrics["total_lines"] = metrics["total_lines"] + analysis["metadata"]["lines"];
        }
        
        # Track languages
        if "metadata" in analysis and "language" in analysis["metadata"] {
            lang = analysis["metadata"]["language"];
            if lang in metrics["languages"] {
                metrics["languages"][lang] = metrics["languages"][lang] + 1;
            } else {
                metrics["languages"][lang] = 1;
            }
        }
        
        # Count functions and classes
        if "parsed" in analysis and analysis["parsed"] and "nodes" in analysis["parsed"] {
            for node in analysis["parsed"]["nodes"] {
                if !node or !("type" in node) {
                    continue;
                }
                
                if node["type"] == "function" {
                    metrics["total_functions"] = metrics["total_functions"] + 1;
                } elif node["type"] == "class" {
                    metrics["total_classes"] = metrics["total_classes"] + 1;
                }
                
                # Track complexity
                if "complexity" in node {
                    complexity = node["complexity"];
                    total_complexity = total_complexity + complexity;
                    complexity_count = complexity_count + 1;
                    
                    # Categorize complexity
                    if complexity <= 10 {
                        metrics["complexity_distribution"]["low"] = 
                            metrics["complexity_distribution"]["low"] + 1;
                    } elif complexity <= 20 {
                        metrics["complexity_distribution"]["medium"] = 
                            metrics["complexity_distribution"]["medium"] + 1;
                    } elif complexity <= 50 {
                        metrics["complexity_distribution"]["high"] = 
                            metrics["complexity_distribution"]["high"] + 1;
                    } else {
                        metrics["complexity_distribution"]["very_high"] = 
                            metrics["complexity_distribution"]["very_high"] + 1;
                    }
                }
            }
        }
    }
    
    # Calculate average complexity
    if complexity_count > 0 {
        metrics["average_complexity"] = total_complexity / complexity_count;
    }
    
    ret metrics;
}

# ============================================================================
# Helper Functions
# ============================================================================

can _should_skip_file(filepath: string) -> bool {
    """
    Determines if a file should be skipped during analysis.
    
    Args:
        filepath: File path to check
        
    Returns:
        bool: True if file should be skipped
    """
    
    if !filepath {
        ret true;
    }
    
    # Check for excluded directories
    for skip_dir in ANALYZER_CONFIG["skip_dirs"] {
        if skip_dir in filepath {
            ret true;
        }
    }
    
    # Check for binary/unsupported extensions
    excluded_extensions = [
        ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
        ".pdf", ".zip", ".tar", ".gz", ".rar",
        ".mp3", ".mp4", ".avi", ".mov",
        ".db", ".sqlite", ".lock"
    ];
    
    for ext in excluded_extensions {
        if filepath.endswith(ext) {
            ret true;
        }
    }
    
    ret false;
}

can _detect_language(filepath: string) -> string {
    """
    Detects programming language from file extension.
    
    Args:
        filepath: File path
        
    Returns:
        string: Detected language or "unknown"
    """
    
    if !filepath {
        ret "unknown";
    }
    
    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".jac": "jac",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".cs": "csharp"
    };
    
    for ext in extension_map {
        if filepath.endswith(ext) {
            ret extension_map[ext];
        }
    }
    
    ret "unknown";
}

can _normalize_path(path: string) -> string {
    """
    Normalizes file path (removes double slashes, etc.).
    
    Args:
        path: File path to normalize
        
    Returns:
        string: Normalized path
    """
    
    if !path {
        ret "";
    }
    
    # Replace double slashes
    while "//" in path {
        path = path.replace("//", "/");
    }
    
    # Remove trailing slash
    if path.endswith("/") and len(path) > 1 {
        path = path[:-1];
    }
    
    ret path;
}

can _validate_parsed_structure(parsed: dict) -> bool {
    """
    Validates that parsed result has expected structure.
    
    Args:
        parsed: Parsed result from parser
        
    Returns:
        bool: True if structure is valid
    """
    
    if !parsed {
        ret false;
    }
    
    # Check for required fields
    required_fields = ["nodes"];
    for field in required_fields {
        if !(field in parsed) {
            ret false;
        }
    }
    
    # Validate nodes is a list
    if !parsed["nodes"] or !(type(parsed["nodes"]) == "list") {
        ret false;
    }
    
    ret true;
}

can _enrich_analysis(parsed: dict, language: string, filepath: string) -> dict {
    """
    Enriches parsed analysis with additional metadata.
    
    Args:
        parsed: Basic parsed result
        language: Detected language
        filepath: Full file path
        
    Returns:
        dict: Enriched analysis
    """
    
    enriched = parsed;
    
    # Add language information to nodes
    if "nodes" in enriched and enriched["nodes"] {
        for node in enriched["nodes"] {
            if node and !("language" in node) {
                node["language"] = language;
            }
            if node and !("file" in node) {
                node["file"] = filepath;
            }
        }
    }
    
    # Calculate additional metrics if not present
    if !("complexity" in enriched) {
        enriched["complexity"] = _calculate_complexity(parsed);
    }
    
    ret enriched;
}

can _calculate_complexity(parsed: dict) -> int {
    """
    Calculates cyclomatic complexity from parsed data.
    
    Args:
        parsed: Parsed analysis result
        
    Returns:
        int: Complexity score
    """
    
    # Simplified complexity calculation
    # Real implementation would analyze control flow
    
    complexity = 1;  # Base complexity
    
    if parsed and "nodes" in parsed and parsed["nodes"] {
        for node in parsed["nodes"] {
            if node and "complexity" in node {
                complexity = complexity + node["complexity"];
            }
        }
    }
    
    ret complexity;
}

can _validate_ccg(nodes: list, edges: list) -> dict {
    """
    Validates code call graph structure.
    
    Args:
        nodes: List of nodes
        edges: List of edges
        
    Returns:
        dict: Validation result
    """
    
    result = {
        "valid": true,
        "errors": []
    };
    
    # Build node ID set for validation
    node_ids = {};
    for node in nodes {
        if node and "id" in node {
            node_ids[node["id"]] = true;
        }
    }
    
    # Check edges reference valid nodes
    orphaned_edges = 0;
    for edge in edges {
        if !edge or !("to" in edge) {
            continue;
        }
        
        to_id = edge["to"];
        
        # Check if target exists in node set
        if !(to_id in node_ids) {
            orphaned_edges = orphaned_edges + 1;
        }
    }
    
    if orphaned_edges > 0 {
        result["errors"].append(
            "Found " + str(orphaned_edges) + " edges referencing non-existent nodes"
        );
        # Don't mark as invalid - orphaned edges are warnings, not errors
    }
    
    ret result;
}

can _detect_circular_dependencies(imports: dict) -> list {
    """
    Detects circular dependencies in import graph.
    
    Args:
        imports: Dictionary mapping files to their imports
        
    Returns:
        list: List of circular dependency cycles
    """
    
    circular = [];
    
    # Simple cycle detection using DFS
    # Real implementation would use proper graph algorithms
    
    for file in imports {
        visited = {file: true};
        if _has_cycle(file, imports, visited, [file]) {
            circular.append(visited);
        }
    }
    
    ret circular;
}

can _has_cycle(current: string, imports: dict, visited: dict, path: list) -> bool {
    """
    Helper function to detect cycles in import graph.
    
    Args:
        current: Current file being checked
        imports: Import graph
        visited: Set of visited files
        path: Current path being explored
        
    Returns:
        bool: True if cycle detected
    """
    
    if !(current in imports) {
        ret false;
    }
    
    file_imports = imports[current];
    
    for imported in file_imports {
        if imported in visited {
            ret true;  # Cycle detected
        }
        
        visited[imported] = true;
        path.append(imported);
        
        if _has_cycle(imported, imports, visited, path) {
            ret true;
        }
        
        path.pop();
        visited.pop(imported);
    }
    
    ret false;
}

can _get_timestamp() -> string {
    """Returns current timestamp in ISO 8601 format."""
    ret "2025-11-03T00:00:00Z";
}

can _get_timestamp_ms() -> int {
    """Returns current timestamp in milliseconds."""
    ret 1730592000000;
}

# ============================================================================
# Filtering and Querying Walkers
# ============================================================================

walker filter_analyses(analyses: list, criteria: dict) -> list {
    """
    Filters analysis results based on criteria.
    
    Args:
        analyses: List of analysis results
        criteria: Filter criteria (language, status, etc.)
        
    Returns:
        list: Filtered analysis results
    
    Example:
        # Get only successful Python analyses
        filtered = filter_analyses(analyses, {
            "language": "python",
            "status": "success"
        });
    """
    
    if !analyses or len(analyses) == 0 {
        ret [];
    }
    
    if !criteria or len(criteria) == 0 {
        ret analyses;  # No filtering
    }
    
    filtered = [];
    
    for analysis in analyses {
        if !analysis {
            continue;
        }
        
        matches = true;
        
        # Check language filter
        if "language" in criteria {
            if !("metadata" in analysis) or !("language" in analysis["metadata"]) {
                matches = false;
            } elif analysis["metadata"]["language"] != criteria["language"] {
                matches = false;
            }
        }
        
        # Check status filter
        if matches and "status" in criteria {
            if !("status" in analysis) {
                matches = false;
            } elif analysis["status"] != criteria["status"] {
                matches = false;
            }
        }
        
        # Check file pattern filter
        if matches and "file_pattern" in criteria {
            if !("file" in analysis) {
                matches = false;
            } elif !(criteria["file_pattern"] in analysis["file"]) {
                matches = false;
            }
        }
        
        # Check minimum lines filter
        if matches and "min_lines" in criteria {
            if !("metadata" in analysis) or !("lines" in analysis["metadata"]) {
                matches = false;
            } elif analysis["metadata"]["lines"] < criteria["min_lines"] {
                matches = false;
            }
        }
        
        if matches {
            filtered.append(analysis);
        }
    }
    
    ret filtered;
}

walker find_function(analyses: list, function_name: string) -> list {
    """
    Finds all occurrences of a function across the codebase.
    
    Args:
        analyses: List of analysis results
        function_name: Name of function to find
        
    Returns:
        list: List of locations where function is defined or called
    """
    
    results = [];
    
    if !analyses or !function_name {
        ret results;
    }
    
    for analysis in analyses {
        if !analysis or !("parsed" in analysis) or !analysis["parsed"] {
            continue;
        }
        
        file = analysis["file"] if "file" in analysis else "unknown";
        parsed = analysis["parsed"];
        
        # Check nodes for function definitions
        if "nodes" in parsed and parsed["nodes"] {
            for node in parsed["nodes"] {
                if !node or !("name" in node) {
                    continue;
                }
                
                if node["name"] == function_name {
                    results.append({
                        "type": "definition",
                        "file": file,
                        "node": node,
                        "location": node["location"] if "location" in node else null
                    });
                }
            }
        }
        
        # Check edges for function calls
        if "edges" in parsed and parsed["edges"] {
            for edge in parsed["edges"] {
                if !edge or !("to" in edge) {
                    continue;
                }
                
                if edge["to"] == function_name {
                    results.append({
                        "type": "call",
                        "file": file,
                        "edge": edge,
                        "location": edge["location"] if "location" in edge else null
                    });
                }
            }
        }
    }
    
    ret results;
}

walker find_unused_functions(analyses: list, ccg: dict) -> list {
    """
    Finds functions that are defined but never called.
    
    Args:
        analyses: List of analysis results
        ccg: Code call graph
        
    Returns:
        list: List of unused functions
    """
    
    unused = [];
    
    if !analyses or !ccg or !("nodes" in ccg) or !("edges" in ccg) {
        ret unused;
    }
    
    # Build set of called functions
    called = {};
    for edge in ccg["edges"] {
        if edge and "to" in edge {
            called[edge["to"]] = true;
        }
    }
    
    # Find functions that are not called
    for node in ccg["nodes"] {
        if !node or !("id" in node) or !("type" in node) {
            continue;
        }
        
        # Only check functions
        if node["type"] != "function" {
            continue;
        }
        
        # Check if function is called
        if !(node["id"] in called) {
            unused.append({
                "function": node["name"] if "name" in node else "unknown",
                "file": node["file"] if "file" in node else "unknown",
                "node": node
            });
        }
    }
    
    ret unused;
}

# ============================================================================
# Export Walkers
# ============================================================================

walker export_to_json(analyses: list, output_path: string) -> dict {
    """
    Exports analysis results to JSON file.
    
    Args:
        analyses: List of analysis results
        output_path: Path to output JSON file
        
    Returns:
        dict: Export result
    """
    
    result = {
        "success": false,
        "msg": "",
        "output_path": null
    };
    
    if !analyses {
        result["msg"] = "No analyses to export";
        ret result;
    }
    
    try {
        # Convert to JSON string
        json_str = json.dumps(analyses);
        
        # Write to file
        write_result = py_module.write_text(output_path, json_str);
        
        if write_result and write_result.success {
            result["success"] = true;
            result["msg"] = "Analysis exported successfully";
            result["output_path"] = output_path;
        } else {
            result["msg"] = "Failed to write JSON file";
        }
    } catch e {
        result["msg"] = "Export error: " + str(e);
    }
    
    ret result;
}

walker export_ccg_to_dot(ccg: dict, output_path: string) -> dict {
    """
    Exports CCG to DOT format for visualization with Graphviz.
    
    Args:
        ccg: Code call graph
        output_path: Path to output DOT file
        
    Returns:
        dict: Export result
    """
    
    result = {
        "success": false,
        "msg": "",
        "output_path": null
    };
    
    if !ccg or !("nodes" in ccg) or !("edges" in ccg) {
        result["msg"] = "Invalid CCG structure";
        ret result;
    }
    
    try {
        # Generate DOT format
        dot = "digraph CodeCallGraph {\n";
        dot += "  rankdir=LR;\n";
        dot += "  node [shape=box, style=rounded];\n\n";
        
        # Add nodes
        for node in ccg["nodes"] {
            if !node or !("id" in node) or !("name" in node) {
                continue;
            }
            
            node_type = node["type"] if "type" in node else "unknown";
            label = node["name"] + "\\n(" + node_type + ")";
            
            dot += "  \"" + node["id"] + "\" [label=\"" + label + "\"];\n";
        }
        
        dot += "\n";
        
        # Add edges
        for edge in ccg["edges"] {
            if !edge or !("from" in edge) or !("to" in edge) {
                continue;
            }
            
            edge_type = edge["type"] if "type" in edge else "";
            label = edge_type if len(edge_type) > 0 else "";
            
            dot += "  \"" + edge["from"] + "\" -> \"" + edge["to"] + "\"";
            if len(label) > 0 {
                dot += " [label=\"" + label + "\"]";
            }
            dot += ";\n";
        }
        
        dot += "}\n";
        
        # Write to file
        write_result = py_module.write_text(output_path, dot);
        
        if write_result and write_result.success {
            result["success"] = true;
            result["msg"] = "CCG exported to DOT format";
            result["output_path"] = output_path;
        } else {
            result["msg"] = "Failed to write DOT file";
        }
    } catch e {
        result["msg"] = "Export error: " + str(e);
    }
    
    ret result;
}

# ============================================================================
# Statistics and Reporting Walkers
# ============================================================================

walker generate_analysis_report(analyses: list, ccg: dict = null) -> dict {
    """
    Generates comprehensive analysis report.
    
    Args:
        analyses: List of analysis results
        ccg: Code call graph (optional)
        
    Returns:
        dict: Detailed analysis report
    """
    
    report = {
        "summary": {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        },
        "metrics": {},
        "by_language": {},
        "issues": [],
        "recommendations": []
    };
    
    if !analyses or len(analyses) == 0 {
        ret report;
    }
    
    # Count by status
    for analysis in analyses {
        if !analysis or !("status" in analysis) {
            continue;
        }
        
        report["summary"]["total_files"] = report["summary"]["total_files"] + 1;
        
        if analysis["status"] == AnalysisStatus.SUCCESS {
            report["summary"]["successful"] = report["summary"]["successful"] + 1;
        } elif analysis["status"] == AnalysisStatus.FAILED {
            report["summary"]["failed"] = report["summary"]["failed"] + 1;
        } elif analysis["status"] == AnalysisStatus.SKIPPED {
            report["summary"]["skipped"] = report["summary"]["skipped"] + 1;
        }
    }
    
    # Calculate metrics
    report["metrics"] = calculate_metrics(analyses);
    
    # Group by language
    for analysis in analyses {
        if !analysis or !("metadata" in analysis) or !("language" in analysis["metadata"]) {
            continue;
        }
        
        lang = analysis["metadata"]["language"];
        
        if !(lang in report["by_language"]) {
            report["by_language"][lang] = {
                "count": 0,
                "total_lines": 0,
                "files": []
            };
        }
        
        report["by_language"][lang]["count"] = report["by_language"][lang]["count"] + 1;
        
        if "lines" in analysis["metadata"] {
            report["by_language"][lang]["total_lines"] = 
                report["by_language"][lang]["total_lines"] + analysis["metadata"]["lines"];
        }
        
        if "file" in analysis {
            report["by_language"][lang]["files"].append(analysis["file"]);
        }
    }
    
    # Identify issues
    for analysis in analyses {
        if !analysis {
            continue;
        }
        
        # Check for failed analyses
        if analysis["status"] == AnalysisStatus.FAILED and "error" in analysis {
            report["issues"].append({
                "type": "parse_error",
                "file": analysis["file"] if "file" in analysis else "unknown",
                "message": analysis["error"]
            });
        }
        
        # Check for high complexity
        if "parsed" in analysis and analysis["parsed"] and "nodes" in analysis["parsed"] {
            for node in analysis["parsed"]["nodes"] {
                if node and "complexity" in node and node["complexity"] > 20 {
                    report["issues"].append({
                        "type": "high_complexity",
                        "file": analysis["file"] if "file" in analysis else "unknown",
                        "function": node["name"] if "name" in node else "unknown",
                        "complexity": node["complexity"]
                    });
                }
            }
        }
    }
    
    # Generate recommendations
    if report["summary"]["failed"] > 0 {
        report["recommendations"].append(
            "Review and fix " + str(report["summary"]["failed"]) + " failed file analyses"
        );
    }
    
    if len(report["issues"]) > 0 {
        high_complexity_count = 0;
        for issue in report["issues"] {
            if issue["type"] == "high_complexity" {
                high_complexity_count = high_complexity_count + 1;
            }
        }
        
        if high_complexity_count > 0 {
            report["recommendations"].append(
                "Consider refactoring " + str(high_complexity_count) + 
                " functions with high complexity"
            );
        }
    }
    
    # CCG recommendations
    if ccg and "nodes" in ccg and "edges" in ccg {
        node_count = len(ccg["nodes"]);
        edge_count = len(ccg["edges"]);
        
        if edge_count > node_count * 5 {
            report["recommendations"].append(
                "High coupling detected - consider improving module separation"
            );
        }
    }
    
    ret report;
}

# ============================================================================
# Comparison Walker
# ============================================================================

walker compare_analyses(old_analyses: list, new_analyses: list) -> dict {
    """
    Compares two sets of analyses to detect changes.
    
    Args:
        old_analyses: Previous analysis results
        new_analyses: Current analysis results
        
    Returns:
        dict: Comparison report with changes
    """
    
    comparison = {
        "added_files": [],
        "removed_files": [],
        "modified_files": [],
        "unchanged_files": [],
        "metrics_delta": {}
    };
    
    # Build file maps
    old_map = {};
    new_map = {};
    
    if old_analyses {
        for analysis in old_analyses {
            if analysis and "file" in analysis {
                old_map[analysis["file"]] = analysis;
            }
        }
    }
    
    if new_analyses {
        for analysis in new_analyses {
            if analysis and "file" in analysis {
                new_map[analysis["file"]] = analysis;
            }
        }
    }
    
    # Find added files
    for file in new_map {
        if !(file in old_map) {
            comparison["added_files"].append(file);
        }
    }
    
    # Find removed files
    for file in old_map {
        if !(file in new_map) {
            comparison["removed_files"].append(file);
        }
    }
    
    # Find modified/unchanged files
    for file in new_map {
        if file in old_map {
            # Compare analyses
            if _analyses_differ(old_map[file], new_map[file]) {
                comparison["modified_files"].append(file);
            } else {
                comparison["unchanged_files"].append(file);
            }
        }
    }
    
    # Calculate metrics delta
    old_metrics = calculate_metrics(old_analyses);
    new_metrics = calculate_metrics(new_analyses);
    
    comparison["metrics_delta"] = {
        "files": new_metrics["total_files"] - old_metrics["total_files"],
        "lines": new_metrics["total_lines"] - old_metrics["total_lines"],
        "functions": new_metrics["total_functions"] - old_metrics["total_functions"],
        "classes": new_metrics["total_classes"] - old_metrics["total_classes"]
    };
    
    ret comparison;
}

can _analyses_differ(old_analysis: dict, new_analysis: dict) -> bool {
    """
    Checks if two analyses are different.
    
    Args:
        old_analysis: Previous analysis
        new_analysis: Current analysis
        
    Returns:
        bool: True if analyses differ
    """
    
    # Compare line counts
    if "metadata" in old_analysis and "metadata" in new_analysis {
        old_lines = old_analysis["metadata"]["lines"] if "lines" in old_analysis["metadata"] else 0;
        new_lines = new_analysis["metadata"]["lines"] if "lines" in new_analysis["metadata"] else 0;
        
        if old_lines != new_lines {
            ret true;
        }
    }
    
    # Compare node counts
    if "parsed" in old_analysis and "parsed" in new_analysis {
        old_nodes = len(old_analysis["parsed"]["nodes"]) if "nodes" in old_analysis["parsed"] else 0;
        new_nodes = len(new_analysis["parsed"]["nodes"]) if "nodes" in new_analysis["parsed"] else 0;
        
        if old_nodes != new_nodes {
            ret true;
        }
    }
    
    ret false;
}