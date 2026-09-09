import os
import requests
from dotenv import load_dotenv

# 1. Load variables from the .env file
load_dotenv()

# 2. Get your exact API key name
api_key = os.getenv("IDEALOGRAM_API_KEY")

if not api_key:
    print("Error: IDEALOGRAM_API_KEY not found in .env file.")
    exit()

# 3. Set up the latest Ideogram 4.0 endpoint
url = "https://api.ideogram.ai/v1/ideogram-v4/generate"
headers = {
    "Api-Key": api_key,
    "Content-Type": "application/json"
}

# 4. Define your prompt 
payload = {
    "text_prompt": "An educational, highly detailed infographic explaining the 'Self-Attention' mechanism in deep learning. Title at the top: 'UNDERSTANDING SELF-ATTENTION IN DEEP LEARNING'. Visual style: clean, modern, tech-themed blue and teal background with circuit board faint patterns. The diagram shows a step-by-step process using the sample sentence 'THE LARGE CAT SAT ON THE MAT'. Step 1: 'Input Embeddings' with arrows pointing to 3D green blocks. Step 2: 'Query, Key, Value Vectors (Q, K, V)' showing arrows splitting into blue, red, and yellow grids. Step 3: A large magnifying glass focusing on the word 'CAT' with lines connecting it to 'LARGE', 'SAT', and 'MAT', labeled 'Attention Mechanism'. Step 4: 'Weighted Sum and Output' showing final combined grids. Include diverse, cartoon-style students at the bottom pointing at and looking up at the diagram. Crisp vector art style, vivid colors, easy to read.",
    "aspect_ratio": "16:9",
    "magic_prompt_option": "AUTO"
}

print("Generating image (this may take a few seconds)...")

# 5. Make the API request
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    # 6. Parse the response and download the image
    response_data = response.json()
    image_url = response_data['data'][0]['url']
    print(f"Success! Image generated at: {image_url}")
    
    # Download the image locally (Links expire, so downloading is required)
    image_response = requests.get(image_url)
    with open("ideogram_result.png", "wb") as file:
        file.write(image_response.content)
    print("Saved to your folder as 'ideogram_result.png'")
    
else:
    print(f"Failed with status code {response.status_code}")
    print(response.text)