import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(token=hf_token)

# Prompt crafted to visually explain self-attention without messy text artifacts
prompt = (
    "A simple and clean educational diagram explaining the self-attention mechanism in a Transformer. "
    "Show the sentence 'The cat sat on the mat' as separate word boxes arranged horizontally. "
    "Highlight the word 'sat' in the center. "
    "Draw arrows from 'sat' to the other words, with thick bright arrows pointing to important related words "
    "such as 'cat' and 'mat', and thin faded arrows pointing to less relevant words. "
    "Arrow thickness visually represents attention strength. "
    "Minimal flat vector infographic style, clean white or light background, "
    "clear spacing, simple geometric shapes, easy to understand, professional AI education illustration, "
    "no complex decorations, no 3D objects, no excessive glowing effects, widescreen 16:9"
)

print("Generating self-attention blog header...")

try:
    image = client.text_to_image(
    prompt,
    model="black-forest-labs/FLUX.1-schnell"
)
    
    filename = "self_attention_concept.jpg"
    image.save(filename)
    print(f"Success! Image saved as {filename}")
    
except Exception as e:
    print(f"Failed to generate: {e}")