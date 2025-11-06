"""
Codebase Genius API Backend
FastAPI server with all required endpoints for the Streamlit frontend
"""

from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse
import logging
import os
import uuid
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""
    APP_NAME = "Codebase Genius API"
    VERSION = "1.0.0"
    API_KEY = os.getenv("API_KEY", "Ea1wGDAmUiwxnFLUJo4vcGPQXNcark9LY9XVtNgORRc")
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Job settings
    MAX_CONCURRENT_JOBS = 5
    JOB_TIMEOUT_SECONDS = 600


# ============================================================================
# Job Management
# ============================================================================

class JobStatus(str, Enum):
    """Job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """Background job tracker"""
    def __init__(self, job_id: str, url: str):
        self.job_id = job_id
        self.url = url
        self.status = JobStatus.QUEUED
        self.progress_percent = 0
        self.current_step = "queued"
        self.message = "Job queued"
        self.result: Optional[Dict[str, Any]] = None
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None


class JobManager:
    """Simple in-memory job manager"""
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.active_count = 0
    
    def create_job(self, url: str) -> Job:
        """Create a new job"""
        job_id = str(uuid.uuid4())
        job = Job(job_id, url)
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} for {url}")
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        step: Optional[str] = None,
        message: Optional[str] = None,
        result: Optional[Dict] = None
    ):
        """Update job status"""
        job = self.get_job(job_id)
        if not job:
            return
        
        if status:
            job.status = status
        if progress is not None:
            job.progress_percent = progress
        if step:
            job.current_step = step
        if message:
            job.message = message
        if result:
            job.result = result
        
        if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
            job.completed_at = datetime.now()
            self.active_count = max(0, self.active_count - 1)


# Global job manager
job_manager = JobManager()


# ============================================================================
# Models
# ============================================================================

class URLValidationRequest(BaseModel):
    """URL validation request"""
    url: str = Field(..., min_length=1, description="Repository URL to validate")


class URLValidationResponse(BaseModel):
    """URL validation response"""
    valid: bool
    message: Optional[str] = None
    source: Optional[str] = None
    normalized_url: Optional[str] = None


class StartAPIRequest(BaseModel):
    """Synchronous documentation generation request"""
    url: str = Field(..., min_length=1, description="Repository URL")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class StartAPIAsyncRequest(BaseModel):
    """Asynchronous documentation generation request"""
    url: str = Field(..., min_length=1, description="Repository URL")


class CheckJobStatusRequest(BaseModel):
    """Job status check request"""
    job_id: str = Field(..., description="Job identifier")


class AnalyzeRequest(BaseModel):
    """Legacy analyze endpoint request"""
    repo_url: str
    repo_name: Optional[str] = None


class GenerateRequest(BaseModel):
    """Legacy generate endpoint request"""
    repo_path: str
    repo_name: str
    readme_summary: Optional[str] = ""
    analyses: list[Dict] = []


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title=Config.APP_NAME,
    version=Config.VERSION,
    description="AI-Powered Documentation Generation API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================================
# Authentication
# ============================================================================

def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Verify API key from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Expected: Bearer <token>"
        )
    
    if parts[1] != Config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return True


# ============================================================================
# URL Validation
# ============================================================================

def validate_repository_url(url: str) -> URLValidationResponse:
    """
    Validate a repository URL
    
    Args:
        url: Repository URL to validate
        
    Returns:
        URLValidationResponse with validation result
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ('http', 'https'):
            return URLValidationResponse(
                valid=False,
                message="URL must use HTTP or HTTPS protocol"
            )
        
        # Check hostname
        if not parsed.netloc:
            return URLValidationResponse(
                valid=False,
                message="Invalid URL: missing hostname"
            )
        
        # Identify source
        hostname = parsed.netloc.lower()
        source = None
        
        if 'github.com' in hostname:
            source = 'GitHub'
        elif 'gitlab.com' in hostname:
            source = 'GitLab'
        elif 'bitbucket.org' in hostname:
            source = 'Bitbucket'
        
        if not source:
            return URLValidationResponse(
                valid=False,
                message="Unsupported repository source. Supported: GitHub, GitLab, Bitbucket"
            )
        
        # Normalize URL (remove trailing slashes, .git)
        normalized = url.rstrip('/')
        if normalized.endswith('.git'):
            normalized = normalized[:-4]
        
        return URLValidationResponse(
            valid=True,
            message=f"Valid {source} repository URL",
            source=source,
            normalized_url=normalized
        )
        
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return URLValidationResponse(
            valid=False,
            message=f"URL validation failed: {str(e)}"
        )


# ============================================================================
# Background Processing
# ============================================================================

async def process_repository_async(job_id: str, url: str):
    """
    Background task to process repository
    
    Args:
        job_id: Job identifier
        url: Repository URL
    """
    try:
        job_manager.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            progress=10,
            step="cloning",
            message="Cloning repository..."
        )
        
        # Simulate cloning
        import asyncio
        await asyncio.sleep(2)
        
        job_manager.update_job(
            job_id,
            progress=40,
            step="analyzing",
            message="Analyzing codebase..."
        )
        
        # Simulate analysis
        await asyncio.sleep(3)
        
        job_manager.update_job(
            job_id,
            progress=70,
            step="generating",
            message="Generating documentation..."
        )
        
        # Simulate generation
        await asyncio.sleep(2)
        
        # Complete job
        result = {
            "success": True,
            "status": "completed",
            "data": {
                "outputs": {
                    "markdown": f"/outputs/docs_{job_id[:8]}/documentation.md",
                    "html": f"/outputs/docs_{job_id[:8]}/documentation.html",
                    "ccg": f"/outputs/docs_{job_id[:8]}/code_call_graph.json"
                }
            },
            "metadata": {
                "processing_time_ms": 7000,
                "files_documented": 42,
                "total_lines": 15234
            },
            "message": "Documentation generated successfully"
        }
        
        job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            step="completed",
            message="Documentation generation completed",
            result=result
        )
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            message=f"Job failed: {str(e)}",
            result={
                "success": False,
                "status": "error",
                "message": str(e)
            }
        )


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": Config.APP_NAME,
        "version": Config.VERSION,
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    """Public health check"""
    return {
        "status": "ok",
        "active_requests": job_manager.active_count,
        "api_version": Config.VERSION,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/walker/version_info")
def version_info():
    """Walker version information (frontend compatibility)"""
    return {
        "walker": "v1.0",
        "api": Config.VERSION,
        "api_version": Config.VERSION,
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Walker Endpoints (Frontend Compatibility)
# ============================================================================

@app.post("/walker/validate_url")
def walker_validate_url(request: URLValidationRequest) -> URLValidationResponse:
    """
    Validate repository URL
    
    This endpoint validates repository URLs without authentication
    for better UX (validation before job submission)
    """
    logger.info(f"Validating URL: {request.url}")
    return validate_repository_url(request.url)


@app.post("/walker/start_api")
def walker_start_api_sync(request: StartAPIRequest):
    """
    Synchronous documentation generation

    Note: This is a placeholder that simulates sync behavior
    For production, implement actual repository processing
    """
    logger.info(f"Sync generation requested for: {request.url}")

    # Validate URL first
    validation = validate_repository_url(request.url)
    if not validation.valid:
        raise HTTPException(
            status_code=400,
            detail=validation.message or "Invalid repository URL"
        )

    # Extract repo name from URL
    import re
    match = re.search(r'/([^/]+/[^/]+)(?:\.git)?$', request.url)
    repo_name = match.group(1).replace('/', '_') if match else "docs"

    # Create output directory
    output_dir = Path("outputs") / repo_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample markdown documentation
    markdown_content = f"""# {repo_name.replace('_', '/')} - Codebase Documentation

## Overview

This documentation was automatically generated for the repository: {request.url}

## Repository Information

- **URL**: {request.url}
- **Source**: {validation.source}
- **Generated**: {datetime.now().isoformat()}

## Code Analysis

### Files Documented: 42
### Total Lines: 15,234

## Project Structure

```
{repo_name}/
├── README.md
├── src/
│   ├── main.py
│   └── utils.py
├── tests/
│   └── test_main.py
└── docs/
    └── documentation.md
```

## Key Components

### Main Module (`main.py`)
The main entry point of the application.

### Utilities (`utils.py`)
Helper functions and utilities.

### Tests (`test_main.py`)
Unit tests for the application.

## Dependencies

- Python 3.8+
- FastAPI
- Streamlit
- Other dependencies as needed

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`

## API Endpoints

- `GET /health` - Health check
- `POST /api/analyze` - Analyze repository
- `POST /api/generate` - Generate documentation

## Generated Files

This documentation includes:
- Markdown documentation (this file)
- HTML documentation (view in browser)
- Code call graph (JSON format)

---
*Generated by Codebase Genius on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Generate sample HTML documentation
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{repo_name.replace('_', '/')} - Documentation</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .content {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .stat {{
            text-align: center;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{repo_name.replace('_', '/')}</h1>
        <p>Codebase Documentation</p>
    </div>

    <div class="content">
        <h2>Overview</h2>
        <p>This documentation was automatically generated for the repository: <code>{request.url}</code></p>

        <div class="stats">
            <div class="stat">
                <h3>42</h3>
                <p>Files Documented</p>
            </div>
            <div class="stat">
                <h3>15,234</h3>
                <p>Total Lines</p>
            </div>
            <div class="stat">
                <h3>5.0s</h3>
                <p>Processing Time</p>
            </div>
        </div>

        <h2>Repository Information</h2>
        <ul>
            <li><strong>URL:</strong> {request.url}</li>
            <li><strong>Source:</strong> {validation.source}</li>
            <li><strong>Generated:</strong> {datetime.now().isoformat()}</li>
        </ul>

        <h2>Project Structure</h2>
        <pre><code>{repo_name}/
├── README.md
├── src/
│   ├── main.py
│   └── utils.py
├── tests/
│   └── test_main.py
└── docs/
    └── documentation.md</code></pre>

        <h2>Key Components</h2>
        <h3>Main Module (<code>main.py</code>)</h3>
        <p>The main entry point of the application.</p>

        <h3>Utilities (<code>utils.py</code>)</h3>
        <p>Helper functions and utilities.</p>

        <h3>Tests (<code>test_main.py</code>)</h3>
        <p>Unit tests for the application.</p>

        <h2>Getting Started</h2>
        <ol>
            <li>Clone the repository</li>
            <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
            <li>Run the application: <code>python main.py</code></li>
        </ol>

        <hr>
        <p style="text-align: center; color: #666;">
            Generated by Codebase Genius on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>"""

    # Generate sample code call graph
    ccg_content = {
        "nodes": [
            {"id": "main.py", "type": "file", "size": 1200},
            {"id": "utils.py", "type": "file", "size": 800},
            {"id": "test_main.py", "type": "file", "size": 600}
        ],
        "links": [
            {"source": "main.py", "target": "utils.py", "type": "import"},
            {"source": "test_main.py", "target": "main.py", "type": "test"}
        ],
        "metadata": {
            "total_files": 42,
            "total_lines": 15234,
            "generated_at": datetime.now().isoformat()
        }
    }

    # Write files to disk
    try:
        # Write markdown
        with open(output_dir / "docs.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Write HTML
        with open(output_dir / "docs.html", 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Write code call graph
        import json
        with open(output_dir / "code_call_graph.json", 'w', encoding='utf-8') as f:
            json.dump(ccg_content, f, indent=2)

        logger.info(f"Generated documentation files in {output_dir}")

    except Exception as e:
        logger.error(f"Failed to write documentation files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate documentation files: {str(e)}"
        )

    # Return result
    result = {
        "success": True,
        "status": "completed",
        "data": {
            "outputs": {
                "markdown": f"/outputs/{repo_name}/docs.md",
                "html": f"/outputs/{repo_name}/docs.html",
                "ccg": f"/outputs/{repo_name}/code_call_graph.json"
            }
        },
        "metadata": {
            "processing_time_ms": 5000,
            "files_documented": 42,
            "total_lines": 15234
        },
        "warnings": [],
        "message": "Documentation generated successfully"
    }

    return result


@app.post("/walker/start_api_async")
async def walker_start_api_async(
    request: StartAPIAsyncRequest,
    background_tasks: BackgroundTasks
):
    """
    Asynchronous documentation generation
    
    Creates a background job and returns job ID for status polling
    """
    logger.info(f"Async generation requested for: {request.url}")
    
    # Validate URL first
    validation = validate_repository_url(request.url)
    if not validation.valid:
        raise HTTPException(
            status_code=400,
            detail=validation.message or "Invalid repository URL"
        )
    
    # Check job limit
    if job_manager.active_count >= Config.MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent jobs ({Config.MAX_CONCURRENT_JOBS}) reached. Please try again later."
        )
    
    # Create job
    job = job_manager.create_job(request.url)
    job_manager.active_count += 1
    
    # Start background processing
    background_tasks.add_task(process_repository_async, job.job_id, request.url)
    
    return {
        "success": True,
        "job_id": job.job_id,
        "message": "Job queued successfully",
        "estimated_completion_seconds": 120,
        "status_check_url": f"/walker/check_job_status"
    }


@app.post("/walker/check_job_status")
def walker_check_job_status(request: CheckJobStatusRequest):
    """
    Check status of async job
    
    Returns current job status and result if completed
    """
    job = job_manager.get_job(request.job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {request.job_id} not found"
        )
    
    response = {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress_percent": job.progress_percent,
        "current_step": job.current_step,
        "message": job.message,
        "completed": job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
    }
    
    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()
        response["processing_time_seconds"] = (
            job.completed_at - job.created_at
        ).total_seconds()
    
    if job.result:
        response["result"] = job.result
    
    return response


# ============================================================================
# Legacy API Endpoints (Backward Compatibility)
# ============================================================================

@app.get("/api/health")
def api_health(authorized: bool = Depends(verify_api_key)):
    """Authenticated health check"""
    return {
        "status": "ok",
        "active_requests": job_manager.active_count,
        "api_version": Config.VERSION,
        "timestamp": datetime.now().isoformat(),
        "authenticated": True
    }


@app.post("/api/analyze")
def analyze_repository(
    request: AnalyzeRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Legacy analyze endpoint"""
    logger.info(f"Legacy analyze called for: {request.repo_url}")
    
    return {
        "success": True,
        "data": {
            "repo_name": request.repo_name or "unknown",
            "analyses": [],
            "ccg": {}
        },
        "msg": "Analysis complete (legacy endpoint)"
    }


@app.post("/api/generate")
def generate_docs(
    request: GenerateRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Legacy generate endpoint"""
    logger.info(f"Legacy generate called for: {request.repo_name}")
    
    return {
        "success": True,
        "data": {
            "outputs": {
                "markdown": f"/outputs/{request.repo_name}/docs.md",
                "html": f"/outputs/{request.repo_name}/docs.html"
            }
        },
        "msg": "Documentation generated (legacy endpoint)"
    }


# ============================================================================
# File Download Endpoints
# ============================================================================

from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pathlib import Path

@app.get("/outputs/{repo_name}/docs.md")
def download_markdown(repo_name: str):
    """Download generated markdown file"""
    file_path = Path("outputs") / repo_name / "docs.md"

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Documentation file not found: {file_path}"
        )

    return FileResponse(
        path=file_path,
        media_type="text/markdown",
        filename=f"{repo_name}_docs.md"
    )


@app.get("/outputs/{repo_name}/docs.html")
def view_html(repo_name: str):
    """View generated HTML documentation in browser"""
    file_path = Path("outputs") / repo_name / "docs.html"

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Documentation file not found: {file_path}"
        )

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Error reading HTML file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read HTML file: {str(e)}"
        )


@app.get("/outputs/{repo_name}/list")
def list_output_files(repo_name: str):
    """List all generated files for a repository"""
    output_dir = Path("outputs") / repo_name

    if not output_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Output directory not found: {output_dir}"
        )

    files = []
    try:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to("outputs")
                files.append({
                    "name": file_path.name,
                    "path": str(relative_path),
                    "size": file_path.stat().st_size,
                    "size_human": _format_file_size(file_path.stat().st_size),
                    "download_url": f"/outputs/{relative_path}",
                    "modified": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat()
                })
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )

    return {
        "repo_name": repo_name,
        "files": files,
        "count": len(files),
        "output_dir": str(output_dir)
    }


@app.get("/outputs/{repo_name}/{filename}")
def download_file(repo_name: str, filename: str):
    """Download any file from repository output directory"""
    file_path = Path("outputs") / repo_name / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}"
        )

    # Prevent directory traversal attacks
    if not file_path.resolve().is_relative_to(Path("outputs").resolve()):
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )

    # Determine media type based on extension
    media_types = {
        ".md": "text/markdown",
        ".html": "text/html",
        ".json": "application/json",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".svg": "image/svg+xml"
    }

    suffix = file_path.suffix.lower()
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name
    )


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


# ============================================================================
# Error Handlers
# ============================================================================

from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "status": "error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "status": "error",
            "message": "Internal server error",
            "status_code": 500
        }
    )


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
