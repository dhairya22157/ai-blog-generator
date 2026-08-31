import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Step 1: Load environment variables from .env file
load_dotenv()

# Step 2: Verify API Key is loaded
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_token:
    print("❌ Error: HUGGINGFACEHUB_API_TOKEN not found in .env file!")
    exit(1)

print("✅ Token loaded successfully. Testing connection to Hugging Face...")

try:
    # Step 3: Initialize the model via Hugging Face Inference API
    # Using a fast instruction-tuned model
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        max_new_tokens=100,
        temperature=0.7,
        huggingfacehub_api_token=api_token
    )
    
    # Wrap in ChatHuggingFace to use as a Chat Model
    chat_model = ChatHuggingFace(llm=llm)

    # Step 4: Invoke the model
    response = chat_model.invoke("in how much parameter does this model have?'")
    
    print("\n--- Response from Hugging Face ---")
    print(response.content)
    print("---------------------------------")
    print("🎉 Success! Your API is ready to use in your project.")

except Exception as e:
    print("\n❌ API call failed!")
    print(f"Error details: {e}")