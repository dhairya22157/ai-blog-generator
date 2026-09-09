import urllib.request
import urllib.parse
import os

# Prompt crafted to visually explain self-attention without messy text artifacts
prompt = (
    "A clean, minimal and beautiful educational infographic diagram explaining the self-attention mechanism in Transformers. "
    "Showing word tokens in boxes: 'The', 'cat', 'sat', 'on', 'the', 'mat' horizontally aligned. "
    "The word 'sat' in the center is highlighted with connection lines radiating to 'cat' and 'mat' with high attention weight. "
    "Modern flat vector illustration, high contrast, clean white background, soft cyan and indigo color palette, 16:9 widescreen"
)

def generate_image_pollinations(prompt: str, output_file: str = "self_attention_flux.png") -> str:
    print("Generating image using Pollinations.ai (FLUX.1)...")
    print(f"Prompt: {prompt[:90]}...")
    
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&model=flux&nologo=true&seed=42"
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
            with open(output_file, "wb") as f:
                f.write(data)
        print(f"Success! Image generated and saved to {output_file} ({len(data)} bytes)")
        return output_file
    except Exception as e:
        print(f"Failed to generate image: {e}")
        return ""

if __name__ == "__main__":
    generate_image_pollinations(prompt, "self_attention_flux.png")