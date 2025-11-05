import os
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig


# Load environment variables early
load_dotenv()

# Configure API if key is available
GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def polish_documentation(
    text: str,
    style: str = 'formal technical',
    model_name: str = 'gemini-2.5-pro',
    temperature: float = 0.7,
    max_output_tokens: int = 2048
) -> str:
    """
    Polish and improve documentation text using Google's Gemini model.

    Args:
        text (str): The original documentation text to polish.
        style (str): Desired writing style (e.g., 'formal technical', 'concise', 'engaging').
        model_name (str): Gemini model to use (default: 'gemini-2.5-pro').
        temperature (float): Creativity level for generation (0.0-1.0; default: 0.7).
        max_output_tokens (int): Maximum tokens in the output (default: 2048).

    Returns:
        str: Polished text, or original if API key or generation fails.

    Raises:
        ValueError: If text is empty or model_name is invalid.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    if not GEMINI_API_KEY:
        return text  # Graceful fallback without API access

    try:
        # Validate model (basic check; extend as needed)
        if model_name not in genai.list_models():
            raise ValueError(f"Invalid model name: {model_name}. Available: {[m.name for m in genai.list_models()]}")

        prompt = f"""You are a professional technical writer. Improve and polish the following documentation for clarity, conciseness, and accuracy. Maintain the original meaning while enhancing structure, grammar, and flow. Style: {style}

Documentation:
{text}"""

        model = genai.GenerativeModel(
            model_name,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=0.95,  # Nucleus sampling for diversity
            ),
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
            }  # Basic safety; customize as needed
        )

        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            return text  # Fallback on empty response

    except Exception as e:
        # Log error in production; here, fallback silently
        print(f"Warning: Failed to polish documentation: {e}", file=os.sys.stderr)
        return text