# code_analyzer.jac - Enhanced code analysis with robust error handling
# Fixed for Jaclang 0.8.10

import:py json;
import:py from datetime { datetime }

# Global configuration for code analysis
glob ANALYZER_CONFIG: dict = {
    "max_file_size_mb": 10,
    "timeout_seconds": 30,
    "skip_binary_files": True,
    "parallel_processing": False,
    "max_concurrent_files": 5,
    "supported_languages": ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c", "jac"],
    "skip_dirs": ["node_modules", ".git", "__pycache__", "venv", "env", "dist", "build", ".idea", ".vscode"]
}

# Analysis status enum
enum AnalysisStatus {
    SUCCESS: str = "success",
    PARTIAL: str = "partial",
    FAILED: str = "failed",
    SKIPPED: str = "skipped",
    TIMEOUT: str = "timeout"
}

# ============================================================================
# Main Analysis Walker - Batch File Processing
# ============================================================================

walker analyze_files {
    has repo_root: str;
    has files: list;
    
    can process with `root entry {
        """
        Analyzes multiple files in a repository.
        
        Args:
            repo_root: Root directory of the repository
            files: List of relative file paths to analyze
            
        Returns:
            list: Analysis results for each file
        """
        
        report here {
            "event": "analyzer:batch_start",
            "data": {
                "repo_root": self.repo_root,
                "file_count": len(self.files) if self.files else 0
            }
        }
        
        # Validate inputs
        if not self.repo_root or len(self.repo_root) == 0 {
            report here {
                "event": "analyzer:error",
                "data": {"error": "Repository root is required"}
            }
            return []
        }
        
        if not self.files {
            report here {
                "event": "analyzer:warning",
                "data": {"warning": "No files provided"}
            }
            return []
        }
        
        if len(self.files) == 0 {
            report here {
                "event": "analyzer:warning",
                "data": {"warning": "Empty files list"}
            }
            return []
        }
        
        results: list = []
        successful: int = 0
        failed: int = 0
        skipped: int = 0
        
        report here {
            "event": "analyzer:progress",
            "data": {
                "status": "processing",
                "total": len(self.files),
                "completed": 0
            }
        }
        
        # Process each file
        for i in range(len(self.files)) {
            file: str = self.files[i]
            
            # Skip null/invalid entries
            if not file or len(file) == 0 {
                results.append({
                    "file": "unknown",
                    "status": AnalysisStatus.SKIPPED,
                    "parsed": None,
                    "metadata": None,
                    "error": "Empty or null file path"
                })
                skipped += 1
                continue
            }
            
            # Check if file should be skipped
            if self._should_skip_file(file) {
                results.append({
                    "file": file,
                    "status": AnalysisStatus.SKIPPED,
                    "parsed": None,
                    "metadata": {
                        "skip_reason": "Binary or excluded file type"
                    },
                    "error": None
                })
                skipped += 1
                report here {
                    "event": "analyzer:file_skipped",
                    "data": {"file": file}
                }
                continue
            }
            
            # Analyze the file
            try {
                result: dict = self.analyze_single_file(file)
                
                if result and result["status"] == AnalysisStatus.SUCCESS {
                    successful += 1
                } else {
                    failed += 1
                }
                
                results.append(result)
                
            } except Exception as e {
                # Handle unexpected errors
                results.append({
                    "file": file,
                    "status": AnalysisStatus.FAILED,
                    "parsed": None,
                    "metadata": None,
                    "error": f"Exception during analysis: {str(e)}"
                })
                failed += 1
                
                report here {
                    "event": "analyzer:file_error",
                    "data": {
                        "file": file,
                        "error": str(e)
                    }
                }
            }
            
            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == len(self.files) {
                report here {
                    "event": "analyzer:progress",
                    "data": {
                        "status": "processing",
                        "total": len(self.files),
                        "completed": i + 1,
                        "successful": successful,
                        "failed": failed,
                        "skipped": skipped
                    }
                }
            }
        }
        
        report here {
            "event": "analyzer:batch_complete",
            "data": {
                "total": len(self.files),
                "successful": successful,
                "failed": failed,
                "skipped": skipped
            }
        }
        
        return results
    }
    
    can analyze_single_file(relpath: str) -> dict {
        """Analyzes a single file"""
        from pathlib import Path
        
        start_time: int = self._get_timestamp_ms()
        
        report here {
            "event": "analyzer:file_start",
            "data": {
                "file": relpath,
                "repo_root": self.repo_root
            }
        }
        
        # Initialize result structure
        result: dict = {
            "file": relpath,
            "status": AnalysisStatus.FAILED,
            "parsed": None,
            "metadata": None,
            "error": None
        }
        
        # Validate inputs
        if not self.repo_root or len(self.repo_root) == 0 {
            result["error"] = "Repository root is required"
            report here {
                "event": "analyzer:file_error",
                "data": result
            }
            return result
        }
        
        if not relpath or len(relpath) == 0 {
            result["error"] = "File path is required"
            report here {
                "event": "analyzer:file_error",
                "data": result
            }
            return result
        }
        
        # Construct full path
        full_path: str = str(Path(self.repo_root) / relpath)
        
        # Detect language
        language: str = self._detect_language(relpath)
        
        # Check if language is supported
        if language and language not in ANALYZER_CONFIG["supported_languages"] {
            result["status"] = AnalysisStatus.SKIPPED
            result["error"] = f"Unsupported language: {language}"
            result["metadata"] = {
                "language": language,
                "skip_reason": "Unsupported language"
            }
            report here {
                "event": "analyzer:file_skipped",
                "data": result
            }
            return result
        }
        
        # Get file metadata
        try {
            file_path = Path(full_path)
            
            if not file_path.exists() {
                result["error"] = f"File does not exist: {full_path}"
                report here {
                    "event": "analyzer:file_error",
                    "data": result
                }
                return result
            }
            
            file_size: int = file_path.stat().st_size
            
            # Check file size
            if file_size > (ANALYZER_CONFIG["max_file_size_mb"] * 1024 * 1024) {
                result["status"] = AnalysisStatus.SKIPPED
                result["error"] = f"File too large: {file_size} bytes"
                result["metadata"] = {
                    "language": language,
                    "size_bytes": file_size,
                    "skip_reason": "File too large"
                }
                report here {
                    "event": "analyzer:file_skipped",
                    "data": result
                }
                return result
            }
            
            # Read file and count lines
            with open(full_path, 'r', encoding='utf-8') as f {
                lines: int = sum(1 for _ in f)
            }
            
        } except Exception as e {
            result["error"] = f"Error getting file info: {str(e)}"
            report here {
                "event": "analyzer:file_error",
                "data": result
            }
            return result
        }
        
        # Parse the file
        report here {
            "event": "analyzer:parsing",
            "data": {"file": relpath}
        }
        
        parsed: dict = None
        try {
            # Simulate parsing (replace with actual parser)
            parsed = self._parse_file_content(full_path, language)
            
            if not parsed {
                result["error"] = "Parser returned null"
                report here {
                    "event": "analyzer:file_error",
                    "data": result
                }
                return result
            }
            
            # Validate parsed structure
            if not self._validate_parsed_structure(parsed) {
                result["status"] = AnalysisStatus.PARTIAL
                result["error"] = "Invalid or incomplete parse result"
                result["parsed"] = parsed
            } else {
                result["status"] = AnalysisStatus.SUCCESS
                result["parsed"] = parsed
            }
            
        } except Exception as e {
            result["error"] = f"Parse error: {str(e)}"
            report here {
                "event": "analyzer:parse_error",
                "data": {
                    "file": relpath,
                    "error": str(e)
                }
            }
            return result
        }
        
        # Build metadata
        processing_time: int = self._get_timestamp_ms() - start_time
        
        result["metadata"] = {
            "language": language,
            "lines": lines,
            "size_bytes": file_size,
            "analysis_time_ms": processing_time,
            "encoding": "utf-8"
        }
        
        report here {
            "event": "analyzer:file_complete",
            "data": {
                "file": relpath,
                "status": result["status"],
                "analysis_time_ms": processing_time
            }
        }
        
        return result
    }
    
    can _should_skip_file(filepath: str) -> bool {
        """Determines if a file should be skipped"""
        if not filepath {
            return True
        }
        
        # Check for excluded directories
        for skip_dir in ANALYZER_CONFIG["skip_dirs"] {
            if skip_dir in filepath {
                return True
            }
        }
        
        # Check for binary/unsupported extensions
        excluded_extensions: list = [
            ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
            ".pdf", ".zip", ".tar", ".gz", ".rar",
            ".mp3", ".mp4", ".avi", ".mov",
            ".db", ".sqlite", ".lock"
        ]
        
        for ext in excluded_extensions {
            if filepath.endswith(ext) {
                return True
            }
        }
        
        return False
    }
    
    can _detect_language(filepath: str) -> str {
        """Detects programming language from file extension"""
        if not filepath {
            return "unknown"
        }
        
        extension_map: dict = {
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
        }
        
        for ext, lang in extension_map.items() {
            if filepath.endswith(ext) {
                return lang
            }
        }
        
        return "unknown"
    }
    
    can _parse_file_content(filepath: str, language: str) -> dict {
        """Simulates file parsing - replace with actual parser"""
        return {
            "nodes": [],
            "edges": [],
            "imports": [],
            "exports": [],
            "complexity": 1
        }
    }
    
    can _validate_parsed_structure(parsed: dict) -> bool {
        """Validates that parsed result has expected structure"""
        if not parsed {
            return False
        }
        
        # Check for required fields
        required_fields: list = ["nodes"]
        for field in required_fields {
            if field not in parsed {
                return False
            }
        }
        
        # Validate nodes is a list
        if not isinstance(parsed["nodes"], list) {
            return False
        }
        
        return True
    }
    
    can _get_timestamp_ms() -> int {
        """Returns current timestamp in milliseconds"""
        return int(datetime.now().timestamp() * 1000)
    }
}

# ============================================================================
# Code Call Graph (CCG) Builder
# ============================================================================

walker build_ccg {
    has repo_root: str;
    has analyses: list;
    
    can process with `root entry {
        """
        Builds a code call graph from analysis results.
        
        Returns:
            dict: Code call graph with nodes and edges
        """
        
        start_time: int = int(datetime.now().timestamp() * 1000)
        
        report here {
            "event": "analyzer:ccg_start",
            "data": {
                "repo_root": self.repo_root,
                "analysis_count": len(self.analyses) if self.analyses else 0
            }
        }
        
        # Initialize result
        result: dict = {
            "success": False,
            "nodes": [],
            "edges": [],
            "statistics": {
                "total_nodes": 0,
                "total_edges": 0,
                "node_types": {},
                "edge_types": {}
            },
            "errors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Validate inputs
        if not self.repo_root or len(self.repo_root) == 0 {
            result["errors"].append("Repository root is required")
            report here {
                "event": "analyzer:ccg_error",
                "data": result
            }
            return result
        }
        
        if not self.analyses {
            result["errors"].append("Analyses list is required")
            report here {
                "event": "analyzer:ccg_error",
                "data": result
            }
            return result
        }
        
        if len(self.analyses) == 0 {
            report here {
                "event": "analyzer:ccg_warning",
                "data": {"warning": "No analyses provided"}
            }
            result["success"] = True
            return result
        }
        
        # Build node list
        nodes: list = []
        name_map: dict = {}
        
        report here {
            "event": "analyzer:ccg_step",
            "data": {"step": "building_nodes", "status": "started"}
        }
        
        for analysis in self.analyses {
            if not analysis or "parsed" not in analysis or not analysis["parsed"] {
                continue
            }
            
            if "nodes" not in analysis["parsed"] or not analysis["parsed"]["nodes"] {
                continue
            }
            
            file: str = analysis.get("file", "unknown")
            
            for node in analysis["parsed"]["nodes"] {
                if not node {
                    continue
                }
                
                if "name" not in node or "id" not in node {
                    result["errors"].append({
                        "file": file,
                        "error": "Node missing required fields (name or id)"
                    })
                    continue
                }
                
                enriched_node: dict = {
                    "id": node["id"],
                    "name": node["name"],
                    "type": node.get("type", "unknown"),
                    "file": file,
                    "metadata": node.get("metadata", {})
                }
                
                nodes.append(enriched_node)
                name_map[node["name"]] = node["id"]
                
                node_type: str = enriched_node["type"]
                if node_type in result["statistics"]["node_types"] {
                    result["statistics"]["node_types"][node_type] += 1
                } else {
                    result["statistics"]["node_types"][node_type] = 1
                }
            }
        }
        
        result["statistics"]["total_nodes"] = len(nodes)
        
        report here {
            "event": "analyzer:ccg_step",
            "data": {
                "step": "building_nodes",
                "status": "completed",
                "node_count": len(nodes)
            }
        }
        
        # Build edge list
        edges: list = []
        
        report here {
            "event": "analyzer:ccg_step",
            "data": {"step": "building_edges", "status": "started"}
        }
        
        for analysis in self.analyses {
            if not analysis or "parsed" not in analysis or not analysis["parsed"] {
                continue
            }
            
            if "edges" not in analysis["parsed"] or not analysis["parsed"]["edges"] {
                continue
            }
            
            file: str = analysis.get("file", "unknown")
            
            for edge in analysis["parsed"]["edges"] {
                if not edge or "to" not in edge {
                    continue
                }
                
                to_name: str = edge["to"]
                target_id: str = name_map.get(to_name, to_name)
                
                enriched_edge: dict = {
                    "from": file,
                    "to": target_id,
                    "type": edge.get("type", "unknown"),
                    "file": file,
                    "metadata": edge.get("metadata", {})
                }
                
                if "from" in edge {
                    enriched_edge["from_node_id"] = edge["from"]
                }
                
                edges.append(enriched_edge)
                
                edge_type: str = enriched_edge["type"]
                if edge_type in result["statistics"]["edge_types"] {
                    result["statistics"]["edge_types"][edge_type] += 1
                } else {
                    result["statistics"]["edge_types"][edge_type] = 1
                }
            }
        }
        
        result["statistics"]["total_edges"] = len(edges)
        result["nodes"] = nodes
        result["edges"] = edges
        result["success"] = True
        
        processing_time: int = int(datetime.now().timestamp() * 1000) - start_time
        result["statistics"]["processing_time_ms"] = processing_time
        
        report here {
            "event": "analyzer:ccg_complete",
            "data": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "processing_time_ms": processing_time,
                "has_errors": len(result["errors"]) > 0
            }
        }
        
        return result
    }
}

# ============================================================================
# Metrics Calculator
# ============================================================================

walker calculate_metrics {
    has analyses: list;
    
    can process with `root entry {
        """Calculates code metrics across the codebase"""
        
        metrics: dict = {
            "total_files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "average_complexity": 0.0,
            "complexity_distribution": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "very_high": 0
            },
            "languages": {}
        }
        
        if not self.analyses or len(self.analyses) == 0 {
            return metrics
        }
        
        total_complexity: float = 0.0
        complexity_count: int = 0
        
        for analysis in self.analyses {
            if not analysis or analysis.get("status") != AnalysisStatus.SUCCESS {
                continue
            }
            
            metrics["total_files"] += 1
            
            # Count lines
            if "metadata" in analysis and "lines" in analysis["metadata"] {
                metrics["total_lines"] += analysis["metadata"]["lines"]
            }
            
            # Track languages
            if "metadata" in analysis and "language" in analysis["metadata"] {
                lang: str = analysis["metadata"]["language"]
                if lang in metrics["languages"] {
                    metrics["languages"][lang] += 1
                } else {
                    metrics["languages"][lang] = 1
                }
            }
            
            # Count functions and classes
            if "parsed" in analysis and analysis["parsed"] and "nodes" in analysis["parsed"] {
                for node in analysis["parsed"]["nodes"] {
                    if not node or "type" not in node {
                        continue
                    }
                    
                    if node["type"] == "function" {
                        metrics["total_functions"] += 1
                    } elif node["type"] == "class" {
                        metrics["total_classes"] += 1
                    }
                    
                    # Track complexity
                    if "complexity" in node {
                        complexity: int = node["complexity"]
                        total_complexity += complexity
                        complexity_count += 1
                        
                        if complexity <= 10 {
                            metrics["complexity_distribution"]["low"] += 1
                        } elif complexity <= 20 {
                            metrics["complexity_distribution"]["medium"] += 1
                        } elif complexity <= 50 {
                            metrics["complexity_distribution"]["high"] += 1
                        } else {
                            metrics["complexity_distribution"]["very_high"] += 1
                        }
                    }
                }
            }
        }
        
        # Calculate average complexity
        if complexity_count > 0 {
            metrics["average_complexity"] = total_complexity / complexity_count
        }
        
        return metrics
    }
}

# ============================================================================
# Filter Walker
# ============================================================================

walker filter_analyses {
    has analyses: list;
    has criteria: dict;
    
    can process with `root entry {
        """Filters analysis results based on criteria"""
        
        if not self.analyses or len(self.analyses) == 0 {
            return []
        }
        
        if not self.criteria or len(self.criteria) == 0 {
            return self.analyses
        }
        
        filtered: list = []
        
        for analysis in self.analyses {
            if not analysis {
                continue
            }
            
            matches: bool = True
            
            # Check language filter
            if "language" in self.criteria {
                if "metadata" not in analysis or "language" not in analysis["metadata"] {
                    matches = False
                } elif analysis["metadata"]["language"] != self.criteria["language"] {
                    matches = False
                }
            }
            
            # Check status filter
            if matches and "status" in self.criteria {
                if "status" not in analysis {
                    matches = False
                } elif analysis["status"] != self.criteria["status"] {
                    matches = False
                }
            }
            
            # Check file pattern filter
            if matches and "file_pattern" in self.criteria {
                if "file" not in analysis {
                    matches = False
                } elif self.criteria["file_pattern"] not in analysis["file"] {
                    matches = False
                }
            }
            
            if matches {
                filtered.append(analysis)
            }
        }
        
        return filtered
    }
}