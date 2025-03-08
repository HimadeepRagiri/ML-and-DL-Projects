# Advanced Multi-Source RAG Chatbot

Welcome to the **Advanced Multi-Source RAG Chatbot** project! This is a Retrieval-Augmented Generation (RAG) based chatbot that leverages multiple data sources—PDFs, YouTube transcripts, web searches, and Wikipedia—to provide accurate and context-rich responses. Built with state-of-the-art tools like Mistral-7B, LangChain, and Gradio, this project offers both a command-line interface and a web-based UI for seamless interaction.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Prerequisites](#prerequisites)
5. [Setup Instructions](#setup-instructions)
   - [Install Dependencies](#install-dependencies)
   - [Obtain API Tokens](#obtain-api-tokens)
   - [Generate YouTube Cookies](#generate-youtube-cookies)
6. [Running the Project](#running-the-project)
   - [Console Mode](#console-mode)
   - [Web Interface](#web-interface)
   - [Colab Notebook](#colab-notebook)
7. [Usage Examples](#usage-examples)
8. [Screenshots](#screenshots)
9. [Project Components](#project-components)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview
The Advanced Multi-Source RAG Chatbot is designed to answer queries by retrieving information from multiple sources and generating concise responses using a quantized Mistral-7B model. It integrates document search (PDFs), YouTube transcript extraction, web search (via Tavily), and Wikipedia summaries, making it a versatile tool for research, education, and general knowledge exploration.

---

## Features
- **Multi-Source Retrieval**: Extracts information from uploaded PDFs, YouTube video descriptions, web searches, and Wikipedia.
- **Quantized LLM**: Uses a 4-bit quantized Mistral-7B model for efficient inference on GPUs.
- **Conversation Memory**: Maintains chat history for context-aware responses.
- **Web Interface**: Built with Gradio for an intuitive user experience.
- **Modular Design**: Organized into separate Python files for easy maintenance and scalability.

---

## Directory Structure
```
Advanced_Multi_Source_RAG_Chatbot/
├── requirements.txt              # Project dependencies
├── main.py                       # Console-based inference script
├── model_config.py               # Model loading and configuration
├── data_processing.py            # PDF loading and vector store creation
├── retrieval.py                  # Multi-source retrieval functions
├── generation.py                 # Response generation and memory management
├── app.py                        # Gradio web interface
├── images/                       # Example screenshots
│   ├── pdf_example.png           # Screenshot of PDF-based response
│   ├── text_example.png          # Screenshot of text query response
│   └── youtube_example.png       # Screenshot of YouTube-based response
├── Advanced_Multi_Source_RAG_Chatbot.ipynb  # Colab notebook with full execution
└── README.md                     # Project documentation
```

---

## Prerequisites
- **Python**: Version 3.8 or higher
- **GPU**: Recommended for faster inference (Mistral-7B runs on CUDA)
- **API Tokens**: Hugging Face token and Tavily API key
- **YouTube Cookies**: Required for transcript retrieval
- **Internet Connection**: For web search and YouTube/Wikipedia retrieval

---

## Setup Instructions

### Install Dependencies
1. Clone the repository:
   ```bash
   git clone --no-checkout https://github.com/HimadeepRagiri/ML-and-DL-Projects.git
   cd ML-and-DL-Projects
   git sparse-checkout init --cone
   git sparse-checkout set NLP_Projects/Advanced_Multi_Source_RAG_Chatbot
   cd NLP_Projects/Advanced_Multi_Source_RAG_Chatbot
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Obtain API Tokens
This project requires two API tokens to function fully:

#### Hugging Face Token
1. Sign up or log in to [Hugging Face](https://huggingface.co/).
2. Go to your profile > **Settings** > **Access Tokens**.
3. Generate a new token (e.g., `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
4. Add the token to your environment or script:
   ```python
   from huggingface_hub import login
   token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your token
   login(token)
   ```

#### Tavily API Key
1. Sign up at [Tavily](https://tavily.com/).
2. Get your API key from the dashboard (e.g., `tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
3. Set it as an environment variable:
   ```python
   import os
   os.environ["TAVILY_API_KEY"] = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your key
   ```
   Alternatively, export it in your terminal:
   ```bash
   export TAVILY_API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

> **Important**: Replace the example tokens with your own. Do not share your tokens publicly!

### Generate YouTube Cookies
To retrieve YouTube transcripts (descriptions), you need a `cookies.txt` file:
1. **Install a Browser Extension**:
   - Use a Chrome/Firefox extension like "Get cookies.txt" (e.g., [Chrome Extension](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbejj)).
2. **Export Cookies**:
   - Visit [YouTube](https://www.youtube.com/), log in (optional), and export cookies using the extension.
   - Save the file as `cookies.txt`.
3. **Place the File**:
   - Move `cookies.txt` to the root directory of the project (`Advanced_Multi_Source_RAG_Chatbot/`).
   - The retrieval script expects it at `/content/cookies.txt` in Colab; adjust the path in `retrieval.py` if running locally (e.g., `./cookies.txt`).

> **Note**: Without `cookies.txt`, YouTube transcript retrieval will fail silently or return an error.

---

## Running the Project

### Console Mode
Run the console-based inference:
```bash
python main.py
```
- This executes a sample query: "What is Low Rank Adaptation in the context of machine learning models?"
- Outputs retrieved sources, response, and conversation memory.

### Web Interface
Launch the Gradio web UI:
```bash
python app.py
```
- Open the provided URL (e.g., `http://127.0.0.1:7860`) in your browser.
- Input a query, optionally upload a PDF or provide a YouTube URL, and click "Generate Response".

### Colab Notebook
1. Open `Advanced_Multi_Source_RAG_Chatbot.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `cookies.txt` to the Colab environment:
   ```python
   from google.colab import files
   files.upload()  # Upload cookies.txt
   ```
3. Run all cells to execute the full project, including the Gradio interface.

---

## Usage Examples
1. **Text Query**: "What is Low Rank Adaptation in machine learning?"
   - Enter the query in the text box and click "Generate Response".
2. **PDF Upload**: Upload a PDF about machine learning, then ask a related question.
3. **YouTube URL**: Provide a URL (e.g., `https://www.youtube.com/watch?v=example`) to extract its description for the response.

See [Screenshots](#screenshots) for visual examples.

---

## Screenshots
Here are example outputs from the Gradio interface:

1. **PDF-Based Response**  
   ![PDF Example](images/pdf_example.png)

2. **Text Query Response**  
   ![Text Example](images/text_example.png)

3. **YouTube-Based Response**  
   ![YouTube Example](images/youtube_example.png)

---

## Project Components

### Files
- **`main.py`**: Console-based inference script with a sample query.
- **`model_config.py`**: Loads the Mistral-7B model, tokenizer, embeddings, and Tavily search tool.
- **`data_processing.py`**: Handles PDF loading and FAISS vector store creation.
- **`retrieval.py`**: Retrieves data from PDFs, web, Wikipedia, and YouTube.
- **`generation.py`**: Generates responses using the LLM and manages conversation memory.
- **`app.py`**: Implements the Gradio web interface.

### Technologies
- **Mistral-7B**: 4-bit quantized LLM for response generation.
- **LangChain**: For embeddings, vector stores, and memory management.
- **FAISS**: Vector store for PDF document search.
- **Tavily**: Web search API.
- **Gradio**: Web UI framework.
- **yt-dlp**: YouTube transcript extraction.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows the existing structure and includes appropriate documentation.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
