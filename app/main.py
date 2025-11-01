import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Define the data model for the API input ---
class PromptInput(BaseModel):
    prompt: str
    model_name: str = "qwen:0.5b" # Set a default model

# --- 2. Initialize the FastAPI App ---
app = FastAPI(title="Ollama LLM API")

# --- 3. Initialize Clients ---
# Endpoint for the local Ollama server
LOCAL_Ollama_ENDPOINT = "http://localhost:11434/api/generate"

# Initialize the Ollama Cloud client using the API key from the .env file
try:
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        print("Warning: OLLAMA_API_KEY not found. Cloud models will not work.")
        ollama_cloud_client = None
    else:
        ollama_cloud_client = Client(
            host="https://ollama.com",
            headers={'Authorization': 'Bearer ' + api_key}
        )
except Exception as e:
    print(f"Could not initialize Ollama Cloud client: {e}")
    ollama_cloud_client = None

# --- 4. Define the API Endpoint ---
@app.post("/generate", tags=["Text Generation"])
def generate_text(prompt_input: PromptInput):
    """
    Takes a prompt and a model name, then routes the request to either the
    local Ollama server or the Ollama Cloud API to generate a text completion.
    """
    try:
        if 'cloud' in prompt_input.model_name:
            # --- Handle Cloud Model Request ---
            if not ollama_cloud_client:
                raise HTTPException(status_code=400, detail="Ollama Cloud API key is not configured.")
            
            messages = [{'role': 'user', 'content': prompt_input.prompt}]
            # Remove '-cloud' suffix for the API call
            api_model_name = prompt_input.model_name.replace('-cloud', '')
            response = ollama_cloud_client.chat(model=api_model_name, messages=messages)
            return {"generated_text": response['message']['content']}

        else:
            # --- Handle Local Model Request ---
            data = {
                "model": prompt_input.model_name,
                "prompt": prompt_input.prompt,
                "stream": False
            }
            response = requests.post(LOCAL_Ollama_ENDPOINT, json=data)
            response.raise_for_status()
            return {"generated_text": response.json().get("response", "")}

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to the local Ollama server.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))