import os
import base64

import requests
from dotenv import load_dotenv

load_dotenv()

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")


def query_image(prompt: str, output_path: str = "generated_image.png"):
    """
    Generate an image using Stability AI API.

    Args:
        prompt: Text description of the image to generate
        output_path: Path to save the generated image
    """
    if not STABILITY_API_KEY:
        raise ValueError(
            "STABILITY_API_KEY is missing. Add it to the .env file and restart the script."
        )

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "text_prompts": [{"text": prompt, "weight": 1}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Unable to reach Stability AI API. Check your internet connection. "
            f"Original error: {exc}"
        ) from exc

    if response.status_code == 200:
        data = response.json()
        img_base64 = data["artifacts"][0]["base64"]
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(img_base64))
        print(f"Image saved successfully to {output_path}!")
        return output_path

    error_text = response.text
    raise RuntimeError(
        f"Stability AI API request failed with status {response.status_code}: {error_text}"
    )


# Test call
if __name__ == "__main__":
    try:
        query_image(
            "generate the image for explaning self attention mechanism in transformers, with a cute cat in the background",
        )
    except Exception as e:
        print(f"Image generation failed: {e}")
        print("Check your internet connection, DNS settings, and Hugging Face token.")