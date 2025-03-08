from langchain.memory import ConversationBufferMemory
from model_config import tokenizer, model
from retrieval import retrieve_sources

# Conversation Memory
memory = ConversationBufferMemory(return_messages=True)


def update_memory(query, response):
    """Stores conversation history"""
    memory.save_context({"input": query}, {"output": response})


# Generate Response with LLM
def generate_response(query, vector_db=None, youtube_url=None):

    # Retrieve sources
    sources = retrieve_sources(query, vector_db=vector_db, youtube_url=youtube_url)

    # Construct the prompt
    prompt = f"""
    You are a helpful and intelligent assistant that must answer questions using only the information provided in the sources below.

    --------------------
    Sources:
    {sources}
    --------------------

    Using only the above sources, provide a clear, direct, and concise answer to the following question. If the sources do not contain sufficient information to answer the question, respond with "Insufficient information provided."

    Question: {query}
    """

    # Tokenize and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Get the length of the input prompt
    input_length = inputs['input_ids'].shape[1]

    # Generate the response
    output = model.generate(**inputs, max_new_tokens=200)

    # Extract only the generated tokens (exclude the prompt)
    generated_ids = output[0, input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Update memory with query and clean answer
    update_memory(query, response)

    return response
