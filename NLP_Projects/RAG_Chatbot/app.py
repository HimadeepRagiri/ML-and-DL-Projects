import gradio as gr
import warnings
from src.qa import retriever_qa

# Suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
if __name__ == "__main__":
    print("Launching RAG Chatbot")
    print("Upload a PDF (e.g., the paper provided in the task)")
    print("Suggested query: 'What this paper is talking about?'")
    rag_application.launch(server_name="0.0.0.0", server_port=7860)