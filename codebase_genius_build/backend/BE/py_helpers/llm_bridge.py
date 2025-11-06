import os
import sys
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold


# Load environment variables
load_dotenv()

# Configure API
GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class DocumentationPolisherError(Exception):
    """Base exception for documentation polisher errors."""
    pass


class APIKeyMissingError(DocumentationPolisherError):
    """Raised when GEMINI_API_KEY is not configured."""
    pass


def get_available_models() -> list[str]:
    """
    Retrieve list of available Gemini model names.
    
    Returns:
        List of model name strings.
    """
    try:
        return [model.name for model in genai.list_models() 
                if 'generateContent' in model.supported_generation_methods]
    except Exception:
        return []


def polish_documentation(
    text: str,
    style: str = 'formal technical',
    model_name: str = 'gemini-2.0-flash-exp',
    temperature: float = 0.7,
    max_output_tokens: int = 2048,
    raise_on_error: bool = False
) -> str:
    """
    Polish and improve documentation text using Google's Gemini model.

    Args:
        text: The original documentation text to polish.
        style: Desired writing style (e.g., 'formal technical', 'concise', 'engaging').
        model_name: Gemini model to use (default: 'gemini-2.0-flash-exp').
        temperature: Creativity level for generation (0.0-1.0; default: 0.7).
        max_output_tokens: Maximum tokens in the output (default: 2048).
        raise_on_error: If True, raise exceptions instead of returning original text.

    Returns:
        Polished text, or original text if generation fails (unless raise_on_error=True).

    Raises:
        ValueError: If text is empty or temperature is out of range.
        APIKeyMissingError: If GEMINI_API_KEY is not set (only when raise_on_error=True).
        DocumentationPolisherError: For other processing errors (only when raise_on_error=True).
    """
    # Input validation
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")
    
    if not 0.0 <= temperature <= 1.0:
        raise ValueError(f"Temperature must be between 0.0 and 1.0, got {temperature}")
    
    if max_output_tokens < 1:
        raise ValueError(f"max_output_tokens must be positive, got {max_output_tokens}")

    # Check API key
    if not GEMINI_API_KEY:
        error_msg = "GEMINI_API_KEY environment variable not set"
        if raise_on_error:
            raise APIKeyMissingError(error_msg)
        print(f"Warning: {error_msg}. Returning original text.", file=sys.stderr)
        return text

    try:
        # Build prompt
        prompt = (
            f"You are a professional technical writer. Improve and polish the following "
            f"documentation for clarity, conciseness, and accuracy. Maintain the original "
            f"meaning while enhancing structure, grammar, and flow.\n\n"
            f"Style: {style}\n\n"
            f"Documentation:\n{text}"
        )

        # Configure model
        model = genai.GenerativeModel(
            model_name,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=0.95,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )

        # Generate content
        response = model.generate_content(prompt)
        
        # Extract and validate response
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        
        # Handle blocked or empty responses
        error_msg = "Model returned empty response"
        if hasattr(response, 'prompt_feedback'):
            error_msg += f" (feedback: {response.prompt_feedback})"
        
        if raise_on_error:
            raise DocumentationPolisherError(error_msg)
        
        print(f"Warning: {error_msg}. Returning original text.", file=sys.stderr)
        return text

    except ValueError:
        # Re-raise validation errors
        raise
    except (APIKeyMissingError, DocumentationPolisherError):
        # Re-raise custom errors if raise_on_error=True
        raise
    except Exception as e:
        error_msg = f"Failed to polish documentation: {type(e).__name__}: {e}"
        if raise_on_error:
            raise DocumentationPolisherError(error_msg) from e
        print(f"Warning: {error_msg}. Returning original text.", file=sys.stderr)
        return text


def main():
    """Example usage of the documentation polisher."""
    sample_text = """
    This function do the thing with the data. It take input and make output.
    Use it when you need process information quickly.
    """
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    polished = polish_documentation(
        sample_text,
        style='concise and clear',
        temperature=0.8
    )
    
    print("Polished text:")
    print(polished)


if __name__ == '__main__':
    main()