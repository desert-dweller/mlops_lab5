import gradio as gr
import requests
import os
from fastapi import FastAPI
from ollama import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local Ollama server endpoint
LOCAL_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# Initialize the Ollama Cloud client using the loaded environment variable
try:
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        print("Warning: OLLAMA_API_KEY not found in .env file. Cloud models will not work.")
        ollama_cloud_client = None
    else:
        ollama_cloud_client = Client(
            host="https://ollama.com",
            headers={'Authorization': 'Bearer ' + api_key}
        )
except Exception as e:
    print(f"Could not initialize Ollama Cloud client: {e}")
    ollama_cloud_client = None

def generate_text(prompt: str, model_name: str):
    """Generates text by routing to the correct Ollama service (local or cloud)."""
    if not prompt:
        return "Please enter a prompt."
    
    try:
        if 'cloud' in model_name:
            # --- Use the Ollama Cloud API ---
            if not ollama_cloud_client:
                return "Error: Ollama Cloud client not initialized. Is OLLAMA_API_KEY set in your .env file?"
            
            messages = [{'role': 'user', 'content': prompt}]
            # Remove '-cloud' suffix for the API call
            api_model_name = model_name.replace('-cloud', '')
            response = ollama_cloud_client.chat(model=api_model_name, messages=messages)
            return response['message']['content']
        else:
            # --- Use the Local Ollama Server ---
            data = { "model": model_name, "prompt": prompt, "stream": False }
            response = requests.post(LOCAL_OLLAMA_ENDPOINT, json=data)
            response.raise_for_status()
            return response.json().get("response", "No response found.")
            
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Dropdown to choose between different models
model_dropdown = gr.Dropdown(
    choices=["qwen:0.5b", "gpt-oss:120b-cloud", "my-custom-model"],
    value="qwen:0.5b",
    label="Select Model (Local, Cloud, or Custom GGUF)"
)

# Define the Gradio user interface
gui = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, label="Your Prompt", placeholder="Enter a starting phrase..."), 
        model_dropdown
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Ollama Universal LLM Client",
    description="Select a model to generate text. The app will route the request to the correct service (Local, Cloud, or a Custom GGUF)."
)

# Initialize the FastAPI app and mount the Gradio GUI
app = FastAPI(title="Ollama Universal API")
app = gr.mount_gradio_app(app, gui, path="/")