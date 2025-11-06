"""
Codebase Genius - Enhanced Streamlit Frontend
Production-ready documentation generation interface with real-time progress tracking
"""

import streamlit as st
import requests
import time
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Application configuration with validation"""
    app_title: str = "Codebase Genius"
    app_subtitle: str = "AI-Powered Documentation Generation from GitHub Repositories"
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    request_timeout: int = 600  # 10 minutes
    poll_interval: int = 2  # seconds
    max_retries: int = 3
    max_history: int = 10
    
    @property
    def endpoints(self) -> dict[str, str]:
        """API endpoint paths"""
        return {
            "sync": "/walker/start_api",
            "async": "/walker/start_api_async",
            "status": "/walker/check_job_status",
            "validate": "/walker/validate_url",
            "health": "/health",
            "version": "/walker/version_info"
        }
    
    def get_endpoint_url(self, endpoint: str) -> str:
        """Get full URL for an endpoint"""
        return f"{self.api_base_url}{self.endpoints.get(endpoint, '')}"


class APIStatus(Enum):
    """API health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthInfo:
    """API health information"""
    status: APIStatus
    active_requests: int = 0
    api_version: str = "unknown"
    error_message: Optional[str] = None


@dataclass
class GenerationOptions:
    """Documentation generation options"""
    use_async: bool = False
    generate_ccg: bool = True
    generate_api_docs: bool = True
    generate_architecture: bool = True


@dataclass
class HistoryItem:
    """Generation history item"""
    url: str
    timestamp: str
    status: str
    success: bool
    processing_time: Optional[float] = None


# ============================================================================
# Initialize Session State
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state with defaults"""
    defaults = {
        "history": [],
        "current_job": None,
        "api_available": None,
        "config": Config()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# API Client
# ============================================================================

class APIClient:
    """Centralized API client with error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        timeout: int = 10,
        **kwargs
    ) -> tuple[bool, dict]:
        """
        Make HTTP request with error handling
        
        Returns:
            Tuple[bool, dict]: (success, response_data)
        """
        try:
            url = self.config.get_endpoint_url(endpoint)
            response = self.session.request(
                method,
                url,
                timeout=timeout,
                **kwargs
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {
                    "error": f"HTTP {response.status_code}",
                    "message": response.text
                }
                
        except requests.exceptions.Timeout:
            return False, {"error": "Request timed out"}
        except requests.exceptions.ConnectionError:
            return False, {"error": "Cannot connect to API server"}
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}
        except json.JSONDecodeError:
            return False, {"error": "Invalid JSON response"}
    
    def check_health(self) -> HealthInfo:
        """Check API health status"""
        success, data = self._make_request("GET", "health", timeout=5)
        
        if success:
            return HealthInfo(
                status=APIStatus.HEALTHY,
                active_requests=data.get("active_requests", 0),
                api_version=data.get("api_version", "unknown")
            )
        else:
            return HealthInfo(
                status=APIStatus.UNHEALTHY,
                error_message=data.get("error", "Unknown error")
            )
    
    def validate_url(self, url: str) -> dict:
        """Validate repository URL"""
        success, data = self._make_request(
            "POST",
            "validate",
            json={"url": url},
            timeout=10
        )
        
        if not success:
            return {
                "valid": False,
                "message": data.get("error", "Validation failed")
            }
        
        return data
    
    def generate_docs_sync(self, url: str) -> dict:
        """Generate documentation synchronously"""
        success, data = self._make_request(
            "POST",
            "sync",
            json={"url": url},
            timeout=self.config.request_timeout
        )
        
        if not success:
            return {
                "success": False,
                "message": data.get("error", "Generation failed"),
                "status": "error"
            }
        
        return data
    
    def generate_docs_async(self, url: str) -> dict:
        """Start asynchronous documentation generation"""
        success, data = self._make_request(
            "POST",
            "async",
            json={"url": url},
            timeout=30
        )
        
        if not success:
            return {
                "success": False,
                "message": data.get("error", "Failed to start job")
            }
        
        return data
    
    def check_job_status(self, job_id: str) -> dict:
        """Check async job status"""
        success, data = self._make_request(
            "POST",
            "status",
            json={"job_id": job_id},
            timeout=10
        )
        
        if not success:
            return {
                "completed": False,
                "status": "error",
                "message": data.get("error", "Status check failed")
            }
        
        return data
    
    def get_version(self) -> Optional[dict]:
        """Get API version information"""
        success, data = self._make_request("GET", "version", timeout=5)
        return data if success else None


# ============================================================================
# UI Components
# ============================================================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        config = st.session_state.config
        st.title(f"🚀 {config.app_title}")
        st.markdown(f"**{config.app_subtitle}**")
        st.markdown("---")
    
    @staticmethod
    def render_api_status(client: APIClient) -> bool:
        """Render API health status in sidebar"""
        st.sidebar.markdown("### 🔌 API Status")
        
        health_info = client.check_health()
        is_healthy = health_info.status == APIStatus.HEALTHY
        
        if is_healthy:
            st.sidebar.success("✅ API Connected")
            st.sidebar.caption(f"Active requests: {health_info.active_requests}")
            st.sidebar.caption(f"Version: {health_info.api_version}")
        else:
            st.sidebar.error("❌ API Unavailable")
            if health_info.error_message:
                st.sidebar.caption(health_info.error_message)
        
        st.session_state.api_available = is_healthy
        return is_healthy
    
    @staticmethod
    def render_url_input() -> str:
        """Render URL input field with examples"""
        st.markdown("### 📝 Enter Repository URL")
        
        url = st.text_input(
            "GitHub/GitLab/Bitbucket HTTPS URL",
            placeholder="https://github.com/username/repository",
            help="Enter the HTTPS URL of the repository you want to document",
            key="repo_url"
        )
        
        with st.expander("📚 Example URLs"):
            examples = [
                "https://github.com/torvalds/linux",
                "https://github.com/microsoft/vscode",
                "https://gitlab.com/gitlab-org/gitlab",
                "https://bitbucket.org/atlassian/python-bitbucket"
            ]
            for example in examples:
                st.code(example)
        
        return url
    
    @staticmethod
    def render_options() -> GenerationOptions:
        """Render generation options"""
        st.markdown("### ⚙️ Generation Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_async = st.checkbox(
                "Async Mode",
                value=False,
                help="Use async mode for large repositories (>100MB)"
            )
            
            include_ccg = st.checkbox(
                "Generate Code Call Graph",
                value=True,
                help="Generate visualization of code relationships"
            )
        
        with col2:
            include_api_docs = st.checkbox(
                "Generate API Docs",
                value=True,
                help="Generate API reference documentation"
            )
            
            include_architecture = st.checkbox(
                "Generate Architecture Docs",
                value=True,
                help="Generate architecture overview"
            )
        
        return GenerationOptions(
            use_async=use_async,
            generate_ccg=include_ccg,
            generate_api_docs=include_api_docs,
            generate_architecture=include_architecture
        )
    
    @staticmethod
    def render_validation_result(validation: dict):
        """Render URL validation result"""
        if validation.get("valid"):
            source = validation.get("source", "repository")
            st.success(f"✅ Valid {source} URL")
            
            if "normalized_url" in validation:
                st.info(f"📎 Normalized: `{validation['normalized_url']}`")
        else:
            message = validation.get("message", "Unknown error")
            st.error(f"❌ Invalid URL: {message}")
    
    @staticmethod
    def render_result_success(result: dict):
        """Render successful generation result"""
        st.success("✅ Documentation Generated Successfully!")
        
        # Display metrics
        if "metadata" in result:
            metadata = result["metadata"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                processing_time = metadata.get("processing_time_ms", 0) / 1000
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            with col2:
                files_count = metadata.get("files_documented", 0)
                st.metric("Files Documented", files_count)
            
            with col3:
                total_lines = metadata.get("total_lines", 0)
                st.metric("Total Lines", f"{total_lines:,}")
        
        # Display output paths
        if "data" in result and "outputs" in result["data"]:
            outputs = result["data"]["outputs"]
            
            st.markdown("### 📄 Generated Files")
            
            output_files = [
                ("Markdown", outputs.get("markdown")),
                ("HTML", outputs.get("html")),
                ("Code Call Graph", outputs.get("ccg"))
            ]
            
            for label, path in output_files:
                if path:
                    st.code(f"{label}: {path}")
            
            # Additional files
            if "additional" in outputs:
                with st.expander("📚 Additional Documentation"):
                    for name, path in outputs["additional"].items():
                        st.code(f"{name}: {path}")
        
        # Display warnings
        warnings = result.get("warnings", [])
        if warnings:
            with st.expander(f"⚠️ Warnings ({len(warnings)})"):
                for warning in warnings:
                    warning_text = warning.get("warning", str(warning))
                    st.warning(warning_text)
        
        # Download info
        st.markdown("### 💾 Download Documentation")
        
        # Extract repo name from outputs
        outputs = result.get("data", {}).get("outputs", {})
        markdown_path = outputs.get("markdown", "")
        
        if markdown_path:
            # Extract repo name from path (e.g., /outputs/Generative-AI/docs.md)
            import re
            match = re.search(r'/outputs/([^/]+)/', markdown_path)
            if match:
                repo_name = match.group(1)
                base_url = st.session_state.config.api_base_url
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    markdown_url = f"{base_url}/outputs/{repo_name}/docs.md"
                    st.markdown(f"[📥 Download Markdown]({markdown_url})")
                
                with col2:
                    html_url = f"{base_url}/outputs/{repo_name}/docs.html"
                    st.markdown(f"[🌐 View HTML]({html_url})")
                
                with col3:
                    list_url = f"{base_url}/outputs/{repo_name}/list"
                    st.markdown(f"[📋 List All Files]({list_url})")
                
                st.caption(f"Files are available at: `{base_url}/outputs/{repo_name}/`")
            else:
                st.info("📝 Documentation files are saved on the server. Contact your administrator to retrieve them.")
        else:
            st.info("📝 Documentation files are saved on the server. Contact your administrator to retrieve them.")
    
    @staticmethod
    def render_result_error(result: dict):
        """Render error result"""
        message = result.get("message", "Unknown error")
        st.error(f"❌ Generation Failed: {message}")
        
        # Display detailed errors
        errors = result.get("errors", [])
        if errors:
            with st.expander(f"🔍 Error Details ({len(errors)})"):
                for error in errors:
                    if isinstance(error, dict):
                        st.code(json.dumps(error, indent=2), language="json")
                    else:
                        st.code(str(error))
        
        # Suggestions
        UIComponents._render_error_suggestions(result)
    
    @staticmethod
    def _render_error_suggestions(result: dict):
        """Render error-specific suggestions"""
        st.markdown("### 💡 Suggestions")
        
        status = result.get("status")
        suggestions = {
            "invalid_input": "Check that your URL is correctly formatted and from a supported source (GitHub, GitLab, or Bitbucket)",
            "timeout": "The repository may be too large. Try using Async Mode for better handling of large repositories",
            "rate_limited": None  # Special handling below
        }
        
        if status == "rate_limited":
            errors = result.get("errors", [{}])
            retry_after = errors[0].get("retry_after_seconds", 60) if errors else 60
            st.info(f"Rate limit exceeded. Please wait {retry_after} seconds before trying again")
        elif status in suggestions and suggestions[status]:
            st.info(suggestions[status])
        else:
            st.info("Check the API logs for more details or contact support")
    
    @staticmethod
    def render_async_progress(client: APIClient, job_id: str) -> dict:
        """Render async job progress with polling"""
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        details_container = st.container()
        
        max_polls = 300  # 10 minutes
        poll_count = 0
        config = st.session_state.config
        
        while poll_count < max_polls:
            status = client.check_job_status(job_id)
            
            if status.get("completed"):
                progress_bar.progress(1.0)
                status_text.success("✅ Job Completed!")
                return status.get("result", {})
            
            # Update progress
            progress = status.get("progress_percent", 0) / 100.0
            current_step = status.get("current_step", "processing")
            message = status.get("message", "Processing...")
            
            progress_bar.progress(progress)
            status_text.info(f"⏳ {message}")
            
            # Show details
            with details_container:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Step: {current_step}")
                with col2:
                    st.caption(f"Progress: {int(progress * 100)}%")
            
            time.sleep(config.poll_interval)
            poll_count += 1
        
        # Timeout
        status_text.error("⏱️ Job polling timed out")
        return {
            "success": False,
            "message": "Job status polling timed out",
            "status": "timeout"
        }
    
    @staticmethod
    def render_history():
        """Render generation history in sidebar"""
        history = st.session_state.history
        if not history:
            return
        
        st.sidebar.markdown("### 📜 Recent Generations")
        
        config = st.session_state.config
        recent_items = list(reversed(history[-config.max_history:]))
        
        for item in recent_items:
            url_preview = item["url"][:30] + "..." if len(item["url"]) > 30 else item["url"]
            
            with st.sidebar.expander(f"🔹 {url_preview}"):
                st.caption(f"Time: {item['timestamp']}")
                st.caption(f"Status: {item['status']}")
                
                if item.get("processing_time"):
                    st.caption(f"Duration: {item['processing_time']:.2f}s")
                
                if item['success']:
                    st.success("✅ Success")
                else:
                    st.error("❌ Failed")


# ============================================================================
# History Management
# ============================================================================

def add_to_history(url: str, result: dict):
    """Add generation to history"""
    processing_time = None
    
    if "metadata" in result:
        metadata = result["metadata"]
        processing_time = metadata.get("processing_time_ms", 0) / 1000
    
    history_item = HistoryItem(
        url=url,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status=result.get("status", "unknown"),
        success=result.get("success", False),
        processing_time=processing_time
    )
    
    st.session_state.history.append(asdict(history_item))
    
    # Limit history size
    config = st.session_state.config
    max_size = config.max_history * 2  # Keep more in memory
    if len(st.session_state.history) > max_size:
        st.session_state.history = st.session_state.history[-max_size:]


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="Codebase Genius",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    init_session_state()
    config = st.session_state.config
    client = APIClient(config)
    ui = UIComponents()
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render header
    ui.render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # API status
        api_available = ui.render_api_status(client)
        
        st.markdown("---")
        
        # History
        ui.render_history()
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ About")
        st.caption("Automatically generates comprehensive documentation for repositories using AI-powered code analysis.")
        
        # Version info
        version_info = client.get_version()
        if version_info:
            api_version = version_info.get("api_version", "unknown")
            st.caption(f"API Version: {api_version}")
    
    # Check API availability
    if not api_available:
        st.error("⚠️ Cannot connect to API server. Please ensure the backend is running.")
        st.info(f"Expected endpoint: {config.api_base_url}")
        st.code(
            "# Start the backend with:\n"
            "python -m uvicorn main:app --host 0.0.0.0 --port 8000",
            language="bash"
        )
        return
    
    # URL input
    url = ui.render_url_input()
    
    # Options
    options = ui.render_options()
    
    # Generate button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button(
            "🚀 Generate Documentation",
            use_container_width=True,
            type="primary"
        )
    
    # Generation logic
    if generate_button:
        if not url or not url.strip():
            st.error("❌ Please enter a repository URL")
            return
        
        # Validate URL
        st.markdown("### 🔍 Validating URL...")
        validation = client.validate_url(url)
        ui.render_validation_result(validation)
        
        if not validation.get("valid"):
            return
        
        # Use normalized URL
        normalized_url = validation.get("normalized_url", url)
        
        # Generate documentation
        st.markdown("---")
        st.markdown("### 🔄 Generating Documentation...")
        
        try:
            if options.use_async:
                # Async mode
                st.info("📤 Starting async job...")
                job_info = client.generate_docs_async(normalized_url)
                
                if not job_info.get("success"):
                    error_msg = job_info.get("message", "Unknown error")
                    st.error(f"❌ Failed to start job: {error_msg}")
                    return
                
                job_id = job_info.get("job_id")
                st.success(f"✅ Job queued with ID: `{job_id}`")
                
                estimated_time = job_info.get("estimated_completion_seconds", 120)
                st.info(f"⏱️ Estimated time: {estimated_time} seconds")
                
                # Poll for completion
                result = ui.render_async_progress(client, job_id)
            else:
                # Sync mode
                with st.spinner("⏳ Processing... This may take a few minutes for large repositories"):
                    result = client.generate_docs_sync(normalized_url)
            
            # Display result
            st.markdown("---")
            st.markdown("### 📊 Result")
            
            if result.get("success"):
                ui.render_result_success(result)
            else:
                ui.render_result_error(result)
            
            add_to_history(url, result)
                
        except Exception as e:
            logger.exception("Unexpected error during generation")
            st.error(f"❌ Unexpected error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 2rem;'>
            <p>Made with ❤️ by Codebase Genius Team</p>
            <p>Powered by Jac Language & FastAPI</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
