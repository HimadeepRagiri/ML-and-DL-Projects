from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def document_loader(file):
    """
    Load a PDF file and extract its content.

    Args:
        file: File object from Gradio interface

    Returns:
        List of Document objects
    """
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document


def text_splitter(data):
    """
    Split documents into chunks for processing.

    Args:
        data: List of Document objects

    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks