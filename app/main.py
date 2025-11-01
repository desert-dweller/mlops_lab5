import gradio as gr
from fastapi import FastAPI
from transformers import pipeline

try:
    # Load the model directly into the application using the transformers pipeline
    generator = pipeline("text-generation", model="distilbert/distilgpt2")
    print("LLM pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading LLam pipeline: {e}")
    generator = None

def generate_text(prompt: str):
    """Generates text completion using the self-contained pipeline."""
    if generator is None:
        return "Error: Text generation pipeline is not available."
    if not prompt:
        return "Please enter a prompt."
    try:
        # Generate text using the loaded model
        results = generator(prompt, max_length=50, num_return_sequences=1)
        return results[0]['generated_text']
    except Exception as e:
        return f"An error occurred during text generation: {str(e)}"

# Define the Gradio user interface
gui = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=3, label="Your Prompt", placeholder="Enter a starting phrase..."),
    outputs=gr.Textbox(label="Generated Text"),
    title="Open-Source LLM Text Generation (distilgpt2)",
    description="A self-contained app running a distilled GPT-2 model."
)

# Initialize the FastAPI app and mount the Gradio GUI
app = FastAPI(title="LLM Text Generation API")
app = gr.mount_gradio_app(app, gui, path="/")