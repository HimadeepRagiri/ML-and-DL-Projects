from langchain.chains import RetrievalQA
from src.llm import get_llm
from src.vector_db import retriever


def retriever_qa(file, query):
    """
    Create a QA chain and answer a query based on the provided file.

    Args:
        file: File object from Gradio interface
        query: User's question

    Returns:
        Answer to the query
    """
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )
    response = qa.invoke(query)
    return response['result']