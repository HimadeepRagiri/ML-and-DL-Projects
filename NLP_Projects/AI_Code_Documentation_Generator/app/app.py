import os
import torch
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
from typing import Optional


class DocumentationGenerator:
    def __init__(self, model_dir: str):
        """
        Initialize the Documentation Generator with model and tokenizer.

        Args:
            model_dir (str): Directory containing the fine-tuned model and tokenizer
        """
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Special tokens
        self.code_token = "<CODE>"
        self.doc_token = "<DOC>"
        self.sep_token = "<SEP>"

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> tuple:
        """
        Load and prepare the model and tokenizer.

        Returns:
            tuple: (model, tokenizer)
        """
        # Load tokenizer with special tokens
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)

        # Load and prepare base model
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        base_model.resize_token_embeddings(len(tokenizer))

        # Load fine-tuned adapter weights
        model = PeftModel.from_pretrained(base_model, self.model_dir)
        model.eval()  # Set to evaluation mode
        model.to(self.device)

        return model, tokenizer

    def generate_documentation(self, code: str, max_length: int = 100) -> str:
        """
        Generate documentation for a given code snippet.

        Args:
            code (str): Input code snippet
            max_length (int, optional): Maximum length of generated documentation. Defaults to 100.

        Returns:
            str: Generated documentation
        """
        # Format input with special tokens
        prompt = f"{self.code_token}{code}{self.sep_token}{self.doc_token}"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate documentation
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Extract documentation from generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        doc_start = generated_text.find(self.doc_token) + len(self.doc_token)
        documentation = generated_text[doc_start:].strip()

        return documentation


class GradioInterface:
    def __init__(self, doc_generator: DocumentationGenerator):
        """
        Initialize the Gradio interface.

        Args:
            doc_generator (DocumentationGenerator): Instance of DocumentationGenerator
        """
        self.doc_generator = doc_generator
        self.interface = self._create_interface()

    def _generate_doc_from_input(self, code: str) -> str:
        """
        Wrapper function for documentation generation.

        Args:
            code (str): Input code snippet

        Returns:
            str: Generated documentation
        """
        return self.doc_generator.generate_documentation(code)

    def _create_interface(self) -> gr.Interface:
        """
        Create and configure the Gradio interface.

        Returns:
            gr.Interface: Configured Gradio interface
        """
        return gr.Interface(
            fn=self._generate_doc_from_input,
            inputs=gr.Textbox(
                lines=10,
                label="Input Code",
                placeholder="Paste your Python code here..."
            ),
            outputs=gr.Textbox(label="Generated Documentation"),
            title="Code Documentation Generator",
            description="Enter a Python code snippet to generate its documentation using a fine-tuned GPT-2 model with LoRA."
        )

    def launch(self, share: bool = True, **kwargs):
        """
        Launch the Gradio interface.

        Args:
            share (bool, optional): Whether to create a public link. Defaults to True.
            **kwargs: Additional arguments to pass to gr.Interface.launch()
        """
        self.interface.launch(share=share, **kwargs)


def main():
    # Set model directory
    model_dir = os.getenv(
        "MODEL_DIR",
        "AI_Code_Documentation_Generator/app/trained_model"
    )

    # Initialize documentation generator
    doc_generator = DocumentationGenerator(model_dir)

    # Create and launch Gradio interface
    app = GradioInterface(doc_generator)
    app.launch()


if __name__ == "__main__":
    main()