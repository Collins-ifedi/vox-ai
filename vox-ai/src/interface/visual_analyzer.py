import os
import sys
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union

# --- Third-party Imports ---
try:
    from openai import AsyncOpenAI, APIError
    import docx
    import PyMuPDF
except ImportError as e:
    print(f"Missing required library: {e.name}. Please run 'pip install \"openai>=1.3.0\" python-docx PyMuPDF'")
    sys.exit(1)

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Logger Setup ---
try:
    from utils.logger import get_logger
    logger = get_logger("VisualAnalyzer")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    logger = logging.getLogger("VisualAnalyzer_Fallback")

class VisualAnalyzer:
    """
    Analyzes images and documents (including PDFs) using OpenAI's models.
    This class is designed for asynchronous use within a server environment.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the analyzer with an OpenAI API key.

        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, it will be
                                     fetched from the 'OPENAI_API_KEY' environment variable.

        Raises:
            ValueError: If the API key is not found in the argument or environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables or passed to constructor.")
            raise ValueError("OpenAI API key is required but was not found.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = "gpt-4o"
        logger.info(f"✅ VisualAnalyzer initialized, using model '{self.model}' for analysis.")

    @staticmethod
    def _encode_image_to_base64(image_path: Path) -> Optional[str]:
        """Encodes an image file to a Base64 string for API submission."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            logger.error(f"Could not read image file at {image_path}: {e}")
            return None

    async def _analyze_image(self, file_path: Path, prompt: str) -> str:
        """
        Analyzes a single image using the OpenAI Vision API.
        """
        logger.info(f"Analyzing image '{file_path.name}' with prompt: '{prompt[:50]}...'")
        base64_image = self._encode_image_to_base64(file_path)
        if not base64_image:
            return "Sorry, I could not read the image file. It might be corrupted or in an unsupported format."

        file_extension = file_path.suffix.lower()
        mime_type = f"image/{'jpeg' if file_extension == '.jpg' else file_extension.strip('.')}"

        messages: List[Dict[str, Union[str, List[Dict]]]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=1024
            )
            analysis = response.choices[0].message.content
            logger.info("Image analysis successful.")
            return analysis.strip() if analysis else "I received an empty response from the vision model."
        except APIError as e:
            logger.error(f"OpenAI API error during image analysis: {e}")
            return f"Sorry, I encountered an API error while analyzing the image: {e.body.get('message', 'Unknown Error')}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during image analysis: {e}", exc_info=True)
            return "An unexpected internal error occurred. Please try again later."

    async def _analyze_document(self, file_path: Path, prompt: str) -> str:
        """
        Analyzes content from text, markdown, docx, or PDF files.
        """
        logger.info(f"Analyzing document '{file_path.name}' with prompt: '{prompt[:50]}...'")
        content = ""
        try:
            # MODIFIED: Handle .txt and .md together
            if file_path.suffix.lower() in {'.txt', '.md'}:
                content = file_path.read_text(encoding='utf-8')
            elif file_path.suffix.lower() == '.docx':
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs if para.text])
            # NEW: Handle .pdf files
            elif file_path.suffix.lower() == '.pdf':
                with fitz.open(file_path) as doc:
                    text_parts = [page.get_text() for page in doc]
                    content = "\n".join(text_parts)
            else:
                return "Unsupported document format."

            if not content.strip():
                return "The document appears to be empty."

        except Exception as e:
            logger.error(f"Failed to read or parse document {file_path}: {e}", exc_info=True)
            return f"Sorry, I couldn't read the document. It might be corrupted. Error: {e}"

        system_prompt = "You are an expert document analyst. A user has provided text from a document and a prompt. Analyze the text to answer the user's prompt thoroughly and accurately."
        user_message = f"User Prompt: \"{prompt}\"\n\n--- Document Content ---\n{content[:8000]}\n--- End of Content ---"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1500
            )
            analysis = response.choices[0].message.content
            logger.info("Document analysis successful.")
            return analysis.strip() if analysis else "I received an empty response from the language model."
        except APIError as e:
            logger.error(f"OpenAI API error during document analysis: {e}")
            return f"Sorry, I encountered an API error analyzing the document: {e.body.get('message', 'Unknown Error')}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during document analysis: {e}", exc_info=True)
            return "An unexpected internal error occurred during document analysis."

    async def analyze(self, file_path: Path, prompt: str) -> str:
        """
        Public method to analyze a file by routing to the correct specialized method.
        """
        if not file_path.exists():
            logger.error(f"Analysis failed: File not found at path {file_path}")
            return "Error: The specified file could not be found on the server."

        ext = file_path.suffix.lower()
        image_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        # MODIFIED: Added new document formats
        doc_formats = {'.txt', '.docx', '.pdf', '.md'}

        if ext in image_formats:
            return await self._analyze_image(file_path, prompt)
        elif ext in doc_formats:
            return await self._analyze_document(file_path, prompt)
        else:
            logger.warning(f"Unsupported file type for analysis: '{ext}'")
            return f"Sorry, I cannot analyze '{ext}' files. Please upload a supported image, document (.txt, .docx), or PDF."