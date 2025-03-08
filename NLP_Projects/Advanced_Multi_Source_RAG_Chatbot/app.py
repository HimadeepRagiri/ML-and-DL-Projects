import gradio as gr
import os
from data_processing import load_documents
from generation import generate_response, memory

# Gradio Web Deployment
def gradio_interface(query, youtube_url, pdf):
    """
    This function is the Gradio callback.
    It checks if a PDF is uploaded, loads it into the vector store,
    and then generates a response based on the provided query, YouTube URL, and PDF.
    It also returns the full chat history.
    """
    vector_db = None
    # If a PDF is provided, load it into FAISS
    if pdf is not None:
        # Gradio's File component returns a dict when type="binary"
        if isinstance(pdf, dict):
            pdf_path = pdf.get("name", "uploaded_pdf.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf["data"])
        elif isinstance(pdf, str):
            pdf_path = pdf
        else:
            pdf_path = None

        if pdf_path is not None and os.path.exists(pdf_path):
            vector_db = load_documents(pdf_path)

    # Generate response using the provided inputs
    response = generate_response(query, vector_db=vector_db, youtube_url=youtube_url)

    # Retrieve conversation history
    history = memory.load_memory_variables({})["history"]
    chat_history = ""
    for msg in history:
        # Check for common attributes in stored messages.
        if hasattr(msg, "role"):
            if msg.role == "human":
                chat_history += f"User: {msg.content}\n"
            elif msg.role == "ai":
                chat_history += f"Assistant: {msg.content}\n"
        elif hasattr(msg, "type"):
            if msg.type == "human":
                chat_history += f"User: {msg.content}\n"
            elif msg.type == "ai":
                chat_history += f"Assistant: {msg.content}\n"
        else:
            chat_history += f"{msg}\n"

    return response, chat_history

# Build Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Advanced RAG Chatbot with PDF, YouTube & Web Search")
    with gr.Row():
         query_input = gr.Textbox(label="Enter your question", placeholder="Type your question here...", lines=2)
         youtube_input = gr.Textbox(label="Enter YouTube URL (optional)", placeholder="YouTube URL here...", lines=1)

    pdf_input = gr.File(label="Upload PDF (optional)", file_count="single", type="binary")
    generate_button = gr.Button("Generate Response")
    response_output = gr.Textbox(label="Response", lines=10)
    chat_history_output = gr.Textbox(label="Chat History", lines=10)

    generate_button.click(
        fn=gradio_interface,
        inputs=[query_input, youtube_input, pdf_input],
        outputs=[response_output, chat_history_output]
    )

# launch the interface
if __name__ == "__main__":
    app.launch()
