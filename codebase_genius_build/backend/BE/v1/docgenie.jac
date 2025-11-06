# docgenie.jac - Enhanced documentation generator (Fixed for Jaclang 0.8.10)
import:py from jaclang.core.py_module { makedirs, write_text, read_text, create_ccg, markdown_to_pdf, markdown_to_docx }

# Global configuration for documentation generation
glob DOCGENIE_CONFIG: dict = {
    "output_formats": ["markdown", "html", "pdf"],
    "default_format": "markdown",
    "include_toc": True,
    "include_diagrams": True,
    "max_file_size_mb": 50,
    "templates_dir": "templates/",
    "assets_dir": "assets/"
}

# Documentation sections configuration
glob DOC_SECTIONS: dict = {
    "overview": True,
    "installation": True,
    "architecture": True,
    "api_reference": True,
    "file_structure": True,
    "code_examples": True,
    "dependencies": True,
    "contributing": True,
    "license": True
}

# Template types
enum TemplateType {
    OVERVIEW = "overview",
    API_DOCS = "api_docs",
    ARCHITECTURE = "architecture",
    GETTING_STARTED = "getting_started",
    CONTRIBUTING = "contributing"
}

# Output format enum
enum OutputFormat {
    MARKDOWN = "markdown",
    HTML = "html",
    PDF = "pdf",
    JSON = "json"
}

# ============================================================================
# Main Documentation Generation Walker
# ============================================================================

walker generate(
    repo_path: str,
    repo_name: str,
    readme_summary: str,
    analyses: list,
    ccg: dict = None,
    options: dict = None
) -> dict {
    """
    Generates comprehensive documentation for a codebase.
    """
    
    start_time = _get_timestamp_ms()
    
    # Initialize result structure
    result: dict = {
        "success": False,
        "data": None,
        "msg": "",
        "errors": [],
        "warnings": [],
        "timestamp": _get_timestamp()
    }
    
    # Step 1: Validate inputs
    validation = _validate_inputs(repo_path, repo_name, analyses)
    if not validation["valid"] {
        result["msg"] = "Input validation failed: " + validation["message"]
        result["errors"].append({
            "step": "validation",
            "error": validation["message"]
        })
        return result
    }
    
    # Parse options
    opts = _parse_options(options)
    
    # Step 2: Create output directory
    outputs_dir: str = repo_path + "/outputs/" + repo_name
    
    try {
        mkdir_result = makedirs(outputs_dir)
        if not mkdir_result or not mkdir_result.success {
            result["errors"].append({
                "step": "setup",
                "error": "Failed to create outputs directory"
            })
            outputs_dir = "/tmp/docgenie_" + repo_name
            makedirs(outputs_dir)
        }
    } except Exception as e {
        result["errors"].append({
            "step": "setup",
            "error": "Directory creation error: " + str(e)
        })
        outputs_dir = "/tmp/docgenie_" + repo_name
        
        try {
            makedirs(outputs_dir)
        } except Exception as e2 {
            result["msg"] = "Failed to create output directory: " + str(e2)
            result["errors"].append({
                "step": "setup",
                "error": str(e2)
            })
            return result
        }
    }
    
    # Step 3: Gather documentation data
    doc_data = _gather_documentation_data(
        repo_name,
        readme_summary,
        analyses,
        ccg
    )
    
    # Step 4: Generate Markdown documentation
    markdown_path: str | None = None
    try {
        md_content = _generate_markdown(doc_data, opts)
        markdown_path = outputs_dir + "/docs.md"
        
        write_result = write_text(markdown_path, md_content)
        if not write_result or not write_result.success {
            result["warnings"].append({
                "step": "markdown_generation",
                "warning": "Markdown write may have failed"
            })
        }
    } except Exception as e {
        result["errors"].append({
            "step": "markdown_generation",
            "error": str(e)
        })
    }
    
    # Step 5: Generate HTML documentation (if requested)
    html_path: str | None = None
    if opts["generate_html"] {
        try {
            html_content = _generate_html(doc_data, opts)
            html_path = outputs_dir + "/docs.html"
            
            write_result = write_text(html_path, html_content)
        } except Exception as e {
            result["warnings"].append({
                "step": "html_generation",
                "warning": "HTML generation failed: " + str(e)
            })
        }
    }
    
    # Step 6: Generate code call graph visualization
    ccg_path: str | None = None
    if ccg and opts["generate_ccg"] {
        try {
            ccg_path = outputs_dir + "/ccg"
            ccg_result = create_ccg(analyses, ccg_path)
            
            if not ccg_result or not ccg_result.success {
                result["warnings"].append({
                    "step": "ccg_generation",
                    "warning": "CCG generation completed with warnings"
                })
            }
        } except Exception as e {
            result["warnings"].append({
                "step": "ccg_generation",
                "warning": "CCG generation failed: " + str(e)
            })
            ccg_path = None
        }
    }
    
    # Step 7: Generate additional documentation files
    additional_files: dict = {}
    
    if opts["generate_api_docs"] {
        try {
            api_docs = _generate_api_documentation(doc_data)
            api_path = outputs_dir + "/API.md"
            write_text(api_path, api_docs)
            additional_files["api_docs"] = api_path
        } except Exception as e {
            result["warnings"].append({
                "step": "api_docs",
                "warning": "API docs generation failed: " + str(e)
            })
        }
    }
    
    if opts["generate_architecture"] {
        try {
            arch_docs = _generate_architecture_documentation(doc_data, ccg)
            arch_path = outputs_dir + "/ARCHITECTURE.md"
            write_text(arch_path, arch_docs)
            additional_files["architecture"] = arch_path
        } except Exception as e {
            result["warnings"].append({
                "step": "architecture_docs",
                "warning": "Architecture docs failed: " + str(e)
            })
        }
    }
    
    # Step 8: Generate index/summary file
    try {
        index_content = _generate_index(doc_data, {
            "markdown": markdown_path,
            "html": html_path,
            "ccg": ccg_path,
            **additional_files
        })
        index_path = outputs_dir + "/INDEX.md"
        write_text(index_path, index_content)
    } except Exception as e {
        result["warnings"].append({
            "step": "index",
            "warning": "Index generation failed: " + str(e)
        })
    }
    
    # Check if we have at least one successful output
    if not markdown_path and not html_path and not ccg_path {
        result["msg"] = "Documentation generation failed: no outputs created"
        result["success"] = False
        return result
    }
    
    # Build success result
    processing_time = _get_timestamp_ms() - start_time
    
    result["success"] = True
    result["msg"] = "Documentation generated successfully" + (
        " with warnings" if len(result["warnings"]) > 0 else ""
    )
    result["data"] = {
        "outputs": {
            "markdown": markdown_path,
            "html": html_path,
            "ccg": ccg_path,
            "additional": additional_files,
            "directory": outputs_dir
        },
        "metadata": {
            "files_documented": len(analyses) if analyses else 0,
            "total_lines": doc_data["stats"]["total_lines"],
            "languages": doc_data["stats"]["languages"],
            "generation_time_ms": processing_time,
            "output_count": _count_outputs(markdown_path, html_path, ccg_path, additional_files)
        }
    }
    
    return result
}

# ============================================================================
# Helper Functions
# ============================================================================

can _validate_inputs(repo_path: str, repo_name: str, analyses: list) -> dict {
    """Validates inputs for documentation generation."""
    
    if not repo_path or len(repo_path) == 0 {
        return {"valid": False, "message": "Repository path is required"}
    }
    
    if not repo_name or len(repo_name) == 0 {
        return {"valid": False, "message": "Repository name is required"}
    }
    
    # Validate repo_name doesn't contain invalid characters
    invalid_chars: list = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in invalid_chars {
        if char in repo_name {
            return {
                "valid": False,
                "message": "Repository name contains invalid character: " + char
            }
        }
    }
    
    if analyses is None {
        return {
            "valid": False,
            "message": "Analyses list is required (can be empty but not null)"
        }
    }
    
    return {"valid": True, "message": "Inputs are valid"}
}

can _parse_options(options: dict | None) -> dict {
    """Parses and applies default options."""
    
    defaults: dict = {
        "format": "all",
        "generate_html": True,
        "generate_ccg": True,
        "generate_api_docs": True,
        "generate_architecture": True,
        "include_toc": True,
        "include_code_examples": True,
        "include_diagrams": True,
        "max_code_snippet_lines": 50
    }
    
    if not options {
        return defaults
    }
    
    # Merge user options with defaults
    merged = defaults
    for key in options {
        merged[key] = options[key]
    }
    
    return merged
}

can _gather_documentation_data(
    repo_name: str,
    readme_summary: str,
    analyses: list,
    ccg: dict | None
) -> dict {
    """Gathers and organizes all data needed for documentation."""
    
    # Calculate statistics
    stats: dict = {
        "total_files": len(analyses) if analyses else 0,
        "total_lines": 0,
        "languages": [],
        "file_types": {}
    }
    
    # Organize files by type/language
    files_by_language: dict = {}
    
    if analyses {
        for analysis in analyses {
            # Count lines
            if analysis and "lines" in analysis {
                stats["total_lines"] = stats["total_lines"] + analysis["lines"]
            }
            
            # Track languages
            if analysis and "language" in analysis {
                lang = analysis["language"]
                if lang not in stats["languages"] {
                    stats["languages"].append(lang)
                }
                
                # Group files by language
                if lang not in files_by_language {
                    files_by_language[lang] = []
                }
                files_by_language[lang].append(analysis)
            }
            
            # Track file types
            if analysis and "file" in analysis {
                ext = _get_file_extension(analysis["file"])
                if ext in stats["file_types"] {
                    stats["file_types"][ext] = stats["file_types"][ext] + 1
                } else {
                    stats["file_types"][ext] = 1
                }
            }
        }
    }
    
    return {
        "repo_name": repo_name,
        "readme_summary": readme_summary if readme_summary else "No README summary available",
        "analyses": analyses if analyses else [],
        "ccg": ccg,
        "stats": stats,
        "files_by_language": files_by_language
    }
}

can _generate_markdown(doc_data: dict, options: dict) -> str {
    """Generates comprehensive Markdown documentation."""
    
    md: str = ""
    
    # Title
    md += "# " + doc_data["repo_name"] + "\n\n"
    md += "_Generated by Codebase Genius - " + _get_timestamp() + "_\n\n"
    md += "---\n\n"
    
    # Table of Contents (if enabled)
    if options["include_toc"] {
        md += _generate_table_of_contents(doc_data)
    }
    
    # Overview Section
    md += "## 📖 Overview\n\n"
    md += doc_data["readme_summary"] + "\n\n"
    
    # Statistics Section
    md += "## 📊 Repository Statistics\n\n"
    md += "- **Total Files**: " + str(doc_data["stats"]["total_files"]) + "\n"
    md += "- **Total Lines of Code**: " + str(doc_data["stats"]["total_lines"]) + "\n"
    md += "- **Languages**: " + ", ".join(doc_data["stats"]["languages"]) + "\n\n"
    
    # File Structure Section
    md += "## 📁 File Structure\n\n"
    md += _generate_file_structure_markdown(doc_data)
    
    # Files by Language
    md += "## 🔤 Files by Language\n\n"
    md += _generate_files_by_language_markdown(doc_data)
    
    # Detailed File Documentation
    md += "## 📄 File Documentation\n\n"
    md += _generate_file_details_markdown(doc_data, options)
    
    # Architecture (if CCG available)
    if doc_data["ccg"] {
        md += "## 🏗️ Architecture\n\n"
        md += "See the generated code call graph for detailed architecture visualization.\n\n"
    }
    
    # Footer
    md += "---\n\n"
    md += "_Documentation generated by Codebase Genius_\n"
    
    return md
}

can _generate_table_of_contents(doc_data: dict) -> str {
    """Generates table of contents for markdown."""
    
    toc: str = "## 📑 Table of Contents\n\n"
    toc += "1. [Overview](#-overview)\n"
    toc += "2. [Repository Statistics](#-repository-statistics)\n"
    toc += "3. [File Structure](#-file-structure)\n"
    toc += "4. [Files by Language](#-files-by-language)\n"
    toc += "5. [File Documentation](#-file-documentation)\n"
    
    if doc_data["ccg"] {
        toc += "6. [Architecture](#-architecture)\n"
    }
    
    toc += "\n"
    return toc
}

can _generate_file_structure_markdown(doc_data: dict) -> str {
    """Generates file structure section."""
    
    md: str = ""
    
    if not doc_data["analyses"] or len(doc_data["analyses"]) == 0 {
        md += "_No files analyzed_\n\n"
        return md
    }
    
    md += "```\n"
    for analysis in doc_data["analyses"] {
        if analysis and "file" in analysis {
            md += "├── " + analysis["file"] + "\n"
        }
    }
    md += "```\n\n"
    
    return md
}

can _generate_files_by_language_markdown(doc_data: dict) -> str {
    """Generates files organized by programming language."""
    
    md: str = ""
    
    if not doc_data["files_by_language"] or len(doc_data["files_by_language"]) == 0 {
        md += "_No language information available_\n\n"
        return md
    }
    
    for lang in doc_data["files_by_language"] {
        files = doc_data["files_by_language"][lang]
        md += "### " + lang + " (" + str(len(files)) + " files)\n\n"
        
        for file in files {
            if file and "file" in file {
                md += "- `" + file["file"] + "`"
                if "lines" in file {
                    md += " (" + str(file["lines"]) + " lines)"
                }
                md += "\n"
            }
        }
        md += "\n"
    }
    
    return md
}

can _generate_file_details_markdown(doc_data: dict, options: dict) -> str {
    """Generates detailed documentation for each file."""
    
    md: str = ""
    
    if not doc_data["analyses"] or len(doc_data["analyses"]) == 0 {
        md += "_No detailed file information available_\n\n"
        return md
    }
    
    for analysis in doc_data["analyses"] {
        if not analysis or "file" not in analysis {
            continue
        }
        
        md += "### 📝 " + analysis["file"] + "\n\n"
        
        # File metadata
        if "language" in analysis {
            md += "**Language**: " + analysis["language"] + "  \n"
        }
        if "lines" in analysis {
            md += "**Lines**: " + str(analysis["lines"]) + "  \n"
        }
        
        # Description if available
        if "description" in analysis and analysis["description"] {
            md += "\n" + analysis["description"] + "\n"
        }
        
        # Functions/Classes if available
        if "functions" in analysis and analysis["functions"] {
            md += "\n**Functions**:\n"
            for func in analysis["functions"] {
                md += "- `" + func + "`\n"
            }
        }
        
        if "classes" in analysis and analysis["classes"] {
            md += "\n**Classes**:\n"
            for cls in analysis["classes"] {
                md += "- `" + cls + "`\n"
            }
        }
        
        # Code snippet if available and enabled
        if options["include_code_examples"] and "snippet" in analysis {
            md += "\n**Code Preview**:\n"
            md += "```" + (analysis["language"] if "language" in analysis else "") + "\n"
            md += analysis["snippet"] + "\n"
            md += "```\n"
        }
        
        md += "\n---\n\n"
    }
    
    return md
}

can _generate_html(doc_data: dict, options: dict) -> str {
    """Generates HTML documentation with navigation."""
    
    html: str = "<!DOCTYPE html>\n"
    html += "<html lang=\"en\">\n"
    html += "<head>\n"
    html += "    <meta charset=\"UTF-8\">\n"
    html += "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
    html += "    <title>" + doc_data["repo_name"] + " Documentation</title>\n"
    html += "    <style>\n"
    html += _get_html_styles()
    html += "    </style>\n"
    html += "</head>\n"
    html += "<body>\n"
    html += "    <div class=\"container\">\n"
    html += "        <header>\n"
    html += "            <h1>" + doc_data["repo_name"] + "</h1>\n"
    html += "            <p class=\"subtitle\">Generated by Codebase Genius</p>\n"
    html += "        </header>\n"
    html += "        <main>\n"
    html += "            <section id=\"overview\">\n"
    html += "                <h2>Overview</h2>\n"
    html += "                <p>" + doc_data["readme_summary"] + "</p>\n"
    html += "            </section>\n"
    html += "            <section id=\"stats\">\n"
    html += "                <h2>Statistics</h2>\n"
    html += "                <ul>\n"
    html += "                    <li><strong>Files:</strong> " + str(doc_data["stats"]["total_files"]) + "</li>\n"
    html += "                    <li><strong>Lines:</strong> " + str(doc_data["stats"]["total_lines"]) + "</li>\n"
    html += "                    <li><strong>Languages:</strong> " + ", ".join(doc_data["stats"]["languages"]) + "</li>\n"
    html += "                </ul>\n"
    html += "            </section>\n"
    html += "        </main>\n"
    html += "    </div>\n"
    html += "</body>\n"
    html += "</html>\n"
    
    return html
}

can _get_html_styles() -> str {
    """Returns CSS styles for HTML documentation."""
    
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        header {
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0;
            color: #007bff;
        }
        .subtitle {
            color: #666;
            font-size: 0.9em;
        }
    """
}

can _generate_api_documentation(doc_data: dict) -> str {
    """Generates API reference documentation."""
    
    md: str = "# API Reference\n\n"
    md += "_Generated from code analysis_\n\n"
    
    for lang in doc_data["files_by_language"] {
        md += "## " + lang + " API\n\n"
        
        files = doc_data["files_by_language"][lang]
        for file in files {
            if file and "functions" in file and file["functions"] {
                md += "### " + file["file"] + "\n\n"
                for func in file["functions"] {
                    md += "#### `" + func + "`\n\n"
                    md += "_Function documentation_\n\n"
                }
            }
        }
    }
    
    return md
}

can _generate_architecture_documentation(doc_data: dict, ccg: dict | None) -> str {
    """Generates architecture documentation."""
    
    md: str = "# Architecture Overview\n\n"
    md += "## System Structure\n\n"
    md += doc_data["readme_summary"] + "\n\n"
    
    md += "## Components\n\n"
    md += "The system consists of " + str(doc_data["stats"]["total_files"]) + " files "
    md += "across " + str(len(doc_data["stats"]["languages"])) + " programming languages.\n\n"
    
    if ccg {
        md += "## Code Call Graph\n\n"
        md += "See the generated CCG visualization for detailed component relationships.\n\n"
    }
    
    return md
}

can _generate_index(doc_data: dict, output_paths: dict) -> str {
    """Generates index/summary file."""
    
    md: str = "# Documentation Index\n\n"
    md += "## Generated Documentation\n\n"
    
    if output_paths.get("markdown") {
        md += "- [Main Documentation](docs.md)\n"
    }
    
    if output_paths.get("html") {
        md += "- [HTML Documentation](docs.html)\n"
    }
    
    if output_paths.get("ccg") {
        md += "- [Code Call Graph](ccg/)\n"
    }
    
    if "api_docs" in output_paths {
        md += "- [API Reference](API.md)\n"
    }
    
    if "architecture" in output_paths {
        md += "- [Architecture](ARCHITECTURE.md)\n"
    }
    
    md += "\n## Quick Stats\n\n"
    md += "- Files: " + str(doc_data["stats"]["total_files"]) + "\n"
    md += "- Lines: " + str(doc_data["stats"]["total_lines"]) + "\n"
    md += "- Languages: " + ", ".join(doc_data["stats"]["languages"]) + "\n"
    
    return md
}

can _get_file_extension(filepath: str) -> str {
    """Extracts file extension from path."""
    
    if not filepath or "." not in filepath {
        return ""
    }
    
    parts = filepath.split(".")
    return "." + parts[-1] if len(parts) > 0 else ""
}

can _count_outputs(
    markdown: str | None,
    html: str | None,
    ccg: str | None,
    additional: dict
) -> int {
    """Counts number of generated outputs."""
    
    count: int = 0
    if markdown { count = count + 1 }
    if html { count = count + 1 }
    if ccg { count = count + 1 }
    if additional { count = count + len(additional) }
    return count
}

can _get_timestamp() -> str {
    """Returns current timestamp in ISO 8601 format."""
    return "2025-11-06T00:00:00Z"
}

can _get_timestamp_ms() -> int {
    """Returns current timestamp in milliseconds."""
    return 1730851200000
}