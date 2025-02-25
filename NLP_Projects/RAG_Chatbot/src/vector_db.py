from langchain_community.vectorstores import Chroma
from src.llm import watsonx_embedding
from src.document import document_loader, text_splitter


def vector_database(chunks):
    """
    Create a vector database from document chunks.

    Args:
        chunks: List of document chunks

    Returns:
        Chroma vector database instance
    """
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


def retriever(file):
    """
    Process a file and create a retriever.

    Args:
        file: File object from Gradio interface

    Returns:
        A retriever object for querying
    """
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever