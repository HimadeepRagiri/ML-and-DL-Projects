import gradio as gr
from transformers import BartForConditionalGeneration, BartTokenizer
from inference import generate_headline
from config import Config

# Load the trained model and tokenizer
model_name = "facebook/bart-base"
model = BartForConditionalGeneration.from_pretrained(Config.output_dir)  # Load from training output directory
tokenizer = BartTokenizer.from_pretrained(model_name)

def gradio_generate(article_text):
    """
    Gradio interface function that wraps the generate_headline() function.
    """
    return generate_headline(article_text, model, tokenizer)

# Gradio interface
demo = gr.Interface(
    fn=gradio_generate,
    inputs=gr.Textbox(lines=10, label="Input Article Text"),
    outputs="text",
    title="News Headline Generator",
    description="Enter a news article text to generate a concise headline."
)

# Launch the Gradio demo
demo.launch(share=True)
