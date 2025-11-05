"""
Codebase Genius - Enhanced Streamlit Frontend
Production-ready documentation generation interface with real-time progress tracking
"""

import streamlit as st
import requests
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""
    APP_TITLE = "Codebase Genius"
    APP_SUBTITLE = "AI-Powered Documentation Generation from GitHub Repositories"
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    REQUEST_TIMEOUT = 600  # 10 minutes
    POLL_INTERVAL = 2  # seconds
    MAX_RETRIES = 3
    
    # API endpoints
    ENDPOINTS = {
        "sync": "/walker/start_api",
        "async": "/walker/start_api_async",
        "status": "/walker/check_job_status",
        "validate": "/walker/validate_url",
        "health": "/health",
        "version": "/walker/version_info"
    }

# ============================================================================
# Initialize Session State
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state variables"""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_job" not in st.session_state:
        st.session_state.current_job = None
    if "api_available" not in st.session_state:
        st.session_state.api_available = None
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

# ============================================================================
# API Client Functions
# ============================================================================

def check_api_health() -> Tuple[str, str]:
    """
    Check if API is available and healthy

    Returns:
        Tuple[str, str]: (status, health_info_json)
    """
    try:
        response = requests.get(
            f"{Config.API_BASE_URL}{Config.ENDPOINTS['health']}",
            timeout=5
        )

        if response.status_code == 200:
            return "healthy", json.dumps(response.json())
        else:
            return "unhealthy", json.dumps({"error": f"HTTP {response.status_code}"})
    except Exception as e:
        return "unhealthy", json.dumps({"error": str(e)})

def validate_url(url: str) -> Dict[str, Any]:
    """
    Validate repository URL before processing
    
    Args:
        url: Repository URL to validate
        
    Returns:
        dict: Validation result
    """
    try:
        response = requests.post(
            f"{Config.API_BASE_URL}{Config.ENDPOINTS['validate']}",
            json={"url": url},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "valid": False,
                "message": f"Validation failed: HTTP {response.status_code}"
            }
    except requests.exceptions.Timeout:
        return {
            "valid": False,
            "message": "Validation request timed out"
        }
    except requests.exceptions.ConnectionError:
        return {
            "valid": False,
            "message": "Cannot connect to API server"
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"Validation error: {str(e)}"
        }

def generate_docs_sync(url: str) -> Dict[str, Any]:
    """
    Generate documentation synchronously
    
    Args:
        url: Repository URL
        
    Returns:
        dict: Generation result
    """
    try:
        response = requests.post(
            f"{Config.API_BASE_URL}{Config.ENDPOINTS['sync']}",
            json={"url": url},
            timeout=Config.REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "message": f"API error: HTTP {response.status_code}",
                "status": "error"
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "Request timed out (try async mode for large repos)",
            "status": "timeout"
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": "Cannot connect to API server",
            "status": "error"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "status": "error"
        }

def generate_docs_async(url: str) -> Dict[str, Any]:
    """
    Start asynchronous documentation generation
    
    Args:
        url: Repository URL
        
    Returns:
        dict: Job information
    """
    try:
        response = requests.post(
            f"{Config.API_BASE_URL}{Config.ENDPOINTS['async']}",
            json={"url": url},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "message": f"API error: HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

def check_job_status(job_id: str) -> Dict[str, Any]:
    """
    Check status of async job
    
    Args:
        job_id: Job identifier
        
    Returns:
        dict: Job status
    """
    try:
        response = requests.post(
            f"{Config.API_BASE_URL}{Config.ENDPOINTS['status']}",
            json={"job_id": job_id},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "completed": False,
                "status": "error",
                "message": f"Status check failed: HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "completed": False,
            "status": "error",
            "message": f"Error: {str(e)}"
        }

def get_api_version() -> Optional[Dict]:
    """Get API version information"""
    try:
        response = requests.get(
            f"{Config.API_BASE_URL}{Config.ENDPOINTS['version']}",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# ============================================================================
# UI Components
# ============================================================================

def render_header():
    """Render application header"""
    st.title(f"🚀 {Config.APP_TITLE}")
    st.markdown(f"**{Config.APP_SUBTITLE}**")
    st.markdown("---")

def render_api_status():
    """Render API health status in sidebar"""
    st.sidebar.markdown("### 🔌 API Status")

    status, health_info_json = check_api_health()
    is_healthy = status == "healthy"
    health_info = json.loads(health_info_json) if health_info_json != "{}" else None

    if is_healthy:
        st.sidebar.success("✅ API Connected")
        if health_info:
            st.sidebar.caption(f"Active requests: {health_info.get('active_requests', 0)}")
            api_version = health_info.get('api_version', 'unknown')
            st.sidebar.caption(f"Version: {api_version}")
    else:
        st.sidebar.error("❌ API Unavailable")
        st.sidebar.caption(f"Endpoint: {Config.API_BASE_URL}")

    st.session_state.api_available = is_healthy
    return is_healthy

def render_url_input() -> str:
    """Render URL input field with validation"""
    st.markdown("### 📝 Enter Repository URL")
    
    url = st.text_input(
        "GitHub/GitLab/Bitbucket HTTPS URL",
        placeholder="https://github.com/username/repository",
        help="Enter the HTTPS URL of the repository you want to document",
        key="repo_url"
    )
    
    # Show examples
    with st.expander("📚 Example URLs"):
        st.code("https://github.com/torvalds/linux")
        st.code("https://github.com/microsoft/vscode")
        st.code("https://gitlab.com/gitlab-org/gitlab")
        st.code("https://bitbucket.org/atlassian/python-bitbucket")
    
    return url

def render_options() -> Dict[str, Any]:
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
    
    return {
        "use_async": use_async,
        "options": {
            "generate_ccg": include_ccg,
            "generate_api_docs": include_api_docs,
            "generate_architecture": include_architecture
        }
    }

def render_validation_result(validation: Dict[str, Any]):
    """Render URL validation result"""
    if validation.get("valid"):
        st.success(f"✅ Valid {validation.get('source', 'repository')} URL")
        if "normalized_url" in validation:
            st.info(f"📎 Normalized: `{validation['normalized_url']}`")
    else:
        st.error(f"❌ Invalid URL: {validation.get('message', 'Unknown error')}")

def render_progress_bar(progress: float, status_text: str):
    """Render progress bar with status"""
    st.progress(progress)
    st.caption(status_text)

def render_result_success(result: Dict[str, Any]):
    """Render successful generation result"""
    st.success("✅ Documentation Generated Successfully!")
    
    # Display metadata
    if "metadata" in result:
        metadata = result["metadata"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Processing Time",
                f"{metadata.get('processing_time_ms', 0) / 1000:.2f}s"
            )
        
        with col2:
            st.metric(
                "Files Documented",
                metadata.get('files_documented', 0)
            )
        
        with col3:
            st.metric(
                "Total Lines",
                f"{metadata.get('total_lines', 0):,}"
            )
    
    # Display output paths
    if "data" in result and "outputs" in result["data"]:
        outputs = result["data"]["outputs"]
        
        st.markdown("### 📄 Generated Files")
        
        if outputs.get("markdown"):
            st.code(outputs["markdown"], language=None)
        
        if outputs.get("html"):
            st.code(outputs["html"], language=None)
        
        if outputs.get("ccg"):
            st.code(outputs["ccg"], language=None)
        
        # Additional files
        if "additional" in outputs:
            with st.expander("📚 Additional Documentation"):
                for name, path in outputs["additional"].items():
                    st.code(f"{name}: {path}", language=None)
    
    # Display warnings if any
    if result.get("warnings") and len(result["warnings"]) > 0:
        with st.expander("⚠️ Warnings"):
            for warning in result["warnings"]:
                st.warning(warning.get("warning", str(warning)))
    
    # Download buttons (if file paths are accessible)
    st.markdown("### 💾 Download Documentation")
    st.info("📝 Documentation files are saved on the server. Contact your administrator to retrieve them.")

def render_result_error(result: Dict[str, Any]):
    """Render error result"""
    st.error(f"❌ Generation Failed: {result.get('message', 'Unknown error')}")
    
    # Display detailed errors
    if result.get("errors") and len(result["errors"]) > 0:
        with st.expander("🔍 Error Details"):
            for error in result["errors"]:
                if isinstance(error, dict):
                    st.code(json.dumps(error, indent=2), language="json")
                else:
                    st.code(str(error))
    
    # Suggestions
    st.markdown("### 💡 Suggestions")
    
    if result.get("status") == "invalid_input":
        st.info("Check that your URL is correctly formatted and from a supported source (GitHub, GitLab, or Bitbucket)")
    elif result.get("status") == "timeout":
        st.info("The repository may be too large. Try using Async Mode for better handling of large repositories")
    elif result.get("status") == "rate_limited":
        retry_after = result.get("errors", [{}])[0].get("retry_after_seconds", 60)
        st.info(f"Rate limit exceeded. Please wait {retry_after} seconds before trying again")
    else:
        st.info("Check the API logs for more details or contact support")

def render_async_progress(job_id: str) -> Dict[str, Any]:
    """
    Render async job progress with polling
    
    Args:
        job_id: Job identifier
        
    Returns:
        dict: Final job result
    """
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    
    max_polls = 300  # 10 minutes with 2-second intervals
    poll_count = 0
    
    while poll_count < max_polls:
        status = check_job_status(job_id)
        
        if status.get("completed"):
            progress_placeholder.progress(1.0)
            status_placeholder.success("✅ Job Completed!")
            return status.get("result", {})
        
        # Update progress
        progress = status.get("progress_percent", 0) / 100.0
        current_step = status.get("current_step", "processing")
        
        progress_placeholder.progress(progress)
        status_placeholder.info(f"⏳ {status.get('message', 'Processing...')}")
        
        # Show details
        with details_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Step: {current_step}")
            with col2:
                st.caption(f"Progress: {int(progress * 100)}%")
        
        time.sleep(Config.POLL_INTERVAL)
        poll_count += 1
    
    # Timeout
    status_placeholder.error("⏱️ Job polling timed out")
    return {
        "success": False,
        "message": "Job status polling timed out",
        "status": "timeout"
    }

def render_history():
    """Render generation history in sidebar"""
    if not st.session_state.history:
        return
    
    st.sidebar.markdown("### 📜 Recent Generations")
    
    for idx, item in enumerate(reversed(st.session_state.history[-5:])):
        with st.sidebar.expander(f"🔹 {item['url'][:30]}..."):
            st.caption(f"Time: {item['timestamp']}")
            st.caption(f"Status: {item['status']}")
            if item.get('success'):
                st.success("✅ Success")
            else:
                st.error("❌ Failed")

def add_to_history(url: str, result: Dict[str, Any]):
    """Add generation to history"""
    st.session_state.history.append({
        "url": url,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": result.get("status", "unknown"),
        "success": result.get("success", False),
        "result": result
    })

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
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
        .success-box {
            padding: 1rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .error-box {
            padding: 1rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # API status
        api_available = render_api_status()
        
        st.markdown("---")
        
        # History
        render_history()
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ About")
        st.caption("Codebase Genius automatically generates comprehensive documentation for your repositories using AI-powered code analysis.")
        
        # Version info
        version_info = get_api_version()
        if version_info:
            st.caption(f"API Version: {version_info.get('api_version', 'unknown')}")
    
    # Main content
    if not api_available:
        st.error("⚠️ Cannot connect to API server. Please ensure the backend is running.")
        st.info(f"Expected endpoint: {Config.API_BASE_URL}")
        st.code(f"# Start the backend with:\npython -m uvicorn main:app --host 0.0.0.0 --port 8000", language="bash")
        return
    
    # URL input
    url = render_url_input()
    
    # Options
    options = render_options()
    
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
        if not url:
            st.error("❌ Please enter a repository URL")
            return
        
        # Validate URL first
        st.markdown("### 🔍 Validating URL...")
        validation = validate_url(url)
        render_validation_result(validation)
        
        if not validation.get("valid"):
            return
        
        # Use normalized URL if available
        normalized_url = validation.get("normalized_url", url)
        
        # Generate documentation
        st.markdown("---")
        st.markdown("### 🔄 Generating Documentation...")
        
        try:
            if options["use_async"]:
                # Async mode
                st.info("📤 Starting async job...")
                job_info = generate_docs_async(normalized_url)
                
                if not job_info.get("success"):
                    st.error(f"❌ Failed to start job: {job_info.get('message')}")
                    return
                
                job_id = job_info.get("job_id")
                st.success(f"✅ Job queued with ID: `{job_id}`")
                st.info(f"⏱️ Estimated time: {job_info.get('estimated_completion_seconds', 120)} seconds")
                
                # Poll for completion
                result = render_async_progress(job_id)
            else:
                # Sync mode
                with st.spinner("⏳ Processing... This may take a few minutes for large repositories"):
                    result = generate_docs_sync(normalized_url)
            
            # Display result
            st.markdown("---")
            st.markdown("### 📊 Result")
            
            if result.get("success"):
                render_result_success(result)
                add_to_history(url, result)
            else:
                render_result_error(result)
                add_to_history(url, result)
                
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 2rem;'>
            <p>Made with ❤️ by Codebase Genius Team</p>
            <p>Powered by Jac Language & FastAPI</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# Utility Pages (Optional)
# ============================================================================

def render_settings_page():
    """Render settings page"""
    st.title("⚙️ Settings")
    
    st.markdown("### 🔌 API Configuration")
    
    api_url = st.text_input(
        "API Base URL",
        value=Config.API_BASE_URL,
        help="URL of the Codebase Genius API server"
    )
    
    if st.button("💾 Save Settings"):
        os.environ["API_BASE_URL"] = api_url
        Config.API_BASE_URL = api_url
        st.success("✅ Settings saved!")
        st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("### 🧪 Test Connection")
    
    if st.button("🔍 Test API Connection"):
        status, health_info_json = check_api_health()
        is_healthy = status == "healthy"
        health_info = json.loads(health_info_json) if health_info_json != "{}" else None

        if is_healthy:
            st.success("✅ API is healthy!")
            st.json(health_info)
        else:
            st.error("❌ Cannot connect to API")

def render_docs_page():
    """Render documentation page"""
    st.title("📚 Documentation")
    
    st.markdown("""
    ## How to Use Codebase Genius
    
    ### 1️⃣ Enter Repository URL
    Enter the HTTPS URL of your GitHub, GitLab, or Bitbucket repository.
    
    **Supported formats:**
    - `https://github.com/username/repository`
    - `https://gitlab.com/username/repository`
    - `https://bitbucket.org/username/repository`
    
    ### 2️⃣ Choose Options
    
    **Async Mode**: Recommended for large repositories (>100MB). Processes in the background.
    
    **Code Call Graph**: Generates a visual graph showing code relationships.
    
    **API Docs**: Generates API reference documentation.
    
    **Architecture Docs**: Generates architecture overview documentation.
    
    ### 3️⃣ Generate
    Click the "Generate Documentation" button and wait for processing.
    
    ### 4️⃣ Download
    Once complete, documentation files will be available on the server.
    
    ## Features
    
    - ✅ **Automatic Analysis**: AI-powered code analysis
    - ✅ **Multiple Formats**: Markdown, HTML, PDF support
    - ✅ **Code Call Graphs**: Visual dependency mapping
    - ✅ **API Documentation**: Automatic API reference generation
    - ✅ **Architecture Docs**: System architecture overview
    - ✅ **Real-time Progress**: Track generation progress
    - ✅ **Error Recovery**: Graceful handling of issues
    
    ## Supported Languages
    
    - Python
    - JavaScript/TypeScript
    - Java
    - Go
    - Rust
    - C/C++
    - Jac
    - And more...
    
    ## Tips
    
    💡 **For large repositories**: Use Async Mode to avoid timeouts
    
    💡 **Private repositories**: Ensure the API has access credentials
    
    💡 **Rate limits**: Wait between requests if you hit rate limits
    
    ## Troubleshooting
    
    ### API Connection Failed
    - Ensure the backend server is running
    - Check firewall settings
    - Verify the API URL in settings
    
    ### Generation Timeout
    - Use Async Mode for large repositories
    - Check repository accessibility
    - Ensure repository is not too large (>500MB)
    
    ### Invalid URL
    - Use HTTPS URLs only
    - Ensure repository is public or API has access
    - Check URL format matches supported sources
    """)

# ============================================================================
# Multi-Page Navigation (Optional)
# ============================================================================

def render_navigation():
    """Render page navigation"""
    pages = {
        "🏠 Home": main,
        "⚙️ Settings": render_settings_page,
        "📚 Documentation": render_docs_page
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    
    selection = st.sidebar.radio("Go to", list(pages.keys()), label_visibility="collapsed")
    
    pages[selection]()

# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    # Simple single-page mode
    main()
    
    # For multi-page mode, use:
    # render_navigation()
