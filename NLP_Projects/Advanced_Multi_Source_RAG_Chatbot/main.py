from model_config import load_models
from data_processing import load_documents
from retrieval import retrieve_sources
from generation import generate_response, memory
import os

def inference():
    # Sample PDF path
    pdf_path = "None"
    vector_db = None

    if os.path.exists(pdf_path):
        print("Loading PDF...")
        vector_db = load_documents(pdf_path)
        print("PDF loaded successfully!")
    else:
        print("No PDF found, proceeding without document search.")

    # Example Query
    query = "What is Low Rank Adaptation in the context of machine learning models?"
    youtube_url = None

    print("\nQuery:", query)
    print("YouTube URL:", youtube_url if youtube_url else "None")

    # source retrieval
    print("\nRetrieving sources...")
    sources = retrieve_sources(query, vector_db=vector_db, youtube_url=youtube_url)
    print("Sources retrieved:")
    print(sources)

    # response generation
    print("\nGenerating response...")
    response = generate_response(query, vector_db=vector_db, youtube_url=youtube_url)
    print("\nResponse:")
    print(response)

    # Check memory
    print("\nConversation Memory:")
    memory_content = memory.load_memory_variables({})
    for msg in memory_content["history"]:
        if msg.type == "human":
            print(f"Q: {msg.content}")
        elif msg.type == "ai":
            print(f"A: {msg.content}\n")

if __name__ == "__main__":
    inference()
    