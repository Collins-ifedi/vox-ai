import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import requests
import yaml
# DALL-E 3 integration requires the openai library
# Ensure you have it installed: pip install openai
try:
    from openai import OpenAI, APIError
except ImportError:
    print("OpenAI library not found. Please install it using: pip install openai")
    sys.exit(1)


# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Project-Level Imports & Local Credential Loader ---
try:
    from utils.logger import get_logger
    logger = get_logger("ImageIO")

    def load_credentials() -> Dict:
        """Loads credentials from the project's config file."""
        config_path = Path(PROJECT_ROOT) / "config" / "credentials.yaml"
        if not config_path.is_file():
            logger.warning(f"Credentials file not found: {config_path}")
            return {}
        try:
            with config_path.open('r') as f:
                credentials = yaml.safe_load(f)
                return credentials if isinstance(credentials, dict) else {}
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Error loading credentials file: {e}")
            return {}
except ImportError:
    # Fallback basic logger if the project's logger utility is not found
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    logger = logging.getLogger("ImageIO_Fallback")

    def load_credentials() -> Dict:
        """Dummy credentials loader for fallback."""
        logger.warning("Using dummy credentials loader because project utils were not found.")
        return {}

# --- Configuration ---
def get_openai_api_key() -> Optional[str]:
    """
    Retrieves the OpenAI API key from environment variables or a credentials file.
    Priority:
    1. 'OPENAI_API_KEY' environment variable.
    2. 'credentials.yaml' file under 'openai' -> 'api_key'.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("OpenAI API key loaded from environment variable.")
        return api_key

    credentials = load_credentials()
    # Look for an 'openai' section in the credentials file
    api_key = credentials.get("OPENAI_API_KEY")

    if api_key:
        logger.info("OpenAI API key loaded from credentials.yaml.")
    else:
        logger.error("OpenAI API key not found in environment variables or credentials.yaml.")
    return api_key

# --- Image Generation Class ---
class ImageIO:
    """
    Handles image generation using OpenAI's DALL-E 3 model.
    Maintains backward compatibility with the previous interface.
    """
    def __init__(self, openai_token: Optional[str] = None, timeout: int = 60):
        """
        Initializes the ImageIO service.

        Args:
            openai_token (Optional[str]): OpenAI API token. If None, it will be fetched.
            timeout (int): Timeout in seconds for network requests.
        """
        self.openai_token = openai_token or get_openai_api_key()
        if not self.openai_token:
            raise ValueError("OpenAI API key is required but missing.")

        # Initialize the OpenAI client
        try:
            self.client = OpenAI(api_key=self.openai_token)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"OpenAI client initialization failed. Check your API key. Error: {e}")

        # A requests session is still useful for downloading the image from the URL
        self.timeout = timeout
        self._session = requests.Session()
        logger.info("✅ ImageIO initialized (OpenAI DALL-E 3 for image generation).")

    def generate_image(self, prompt: str, output_file: Path) -> Optional[Path]:
        """
        Generates an image from a text prompt using DALL-E 3 and saves it to a file.

        Args:
            prompt (str): The text prompt to generate the image from.
            output_file (Path): The path where the generated image will be saved.

        Returns:
            Optional[Path]: The path to the saved image file, or None on failure.
        """
        if not prompt.strip():
            logger.warning("⚠️ Image generation skipped: Prompt is empty.")
            return None

        logger.info(f"🎨 Generating image with DALL-E 3. Prompt: '{prompt[:80]}...'")

        try:
            # --- Call DALL-E 3 API ---
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024", # DALL-E 3 supports 1024x1024, 1792x1024, or 1024x1792
                quality="standard", # 'standard' or 'hd'
                response_format="url", # 'url' or 'b64_json'
            )

            # --- Download the image from the returned URL ---
            image_url = response.data[0].url
            if not image_url:
                logger.error("❌ DALL-E 3 API did not return an image URL.")
                return None

            logger.info(f"Image generated, downloading from URL...")
            image_response = self._session.get(image_url, timeout=self.timeout)
            image_response.raise_for_status() # Raise an exception for bad status codes

            # --- Save the image content to the specified file ---
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "wb") as f:
                f.write(image_response.content)

            logger.info(f"✅ Image saved successfully: {output_file}")
            return output_file

        except APIError as e:
            logger.error(f"❌ OpenAI API error during image generation: {e}")
            # Log more details if available
            if e.response and e.response.text:
                 logger.error(f"API Response Body: {e.response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error downloading generated image: {e}")
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during image generation: {e}")

        return None

    def close(self):
        """Closes the underlying HTTP session."""
        self._session.close()
        logger.info("ImageIO HTTP session closed.")


def main():
    """
    Simple test for ImageIO image generation using DALL-E 3.
    """
    print("--- DALL-E 3 Image Generation Test ---")
    image_io = None
    try:
        # Initialize the ImageIO class which now uses DALL-E 3
        image_io = ImageIO()
        output_dir = Path("./generated_media")
        output_dir.mkdir(exist_ok=True)

        # Ask user for a prompt
        user_prompt = input("\nEnter a prompt for DALL-E 3 image generation: ").strip()
        if not user_prompt:
            print("\n⚠️ No prompt entered. Exiting.")
            return

        # Generate image
        image_output_path = output_dir / "dalle3_generated_image.png"
        print(f"\nGenerating image, please wait... This can take up to a minute.")
        generated_image = image_io.generate_image(user_prompt, image_output_path)

        if generated_image:
            print(f"\n✅ Image generated successfully: {generated_image.resolve()}")
        else:
            print("\n❌ Image generation failed. Please check the logs for details.")

    except ValueError as e:
        print(f"\nInitialization Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if image_io:
            image_io.close()
        print("\n--- Test Complete ---")


if __name__ == "__main__":
    main()
