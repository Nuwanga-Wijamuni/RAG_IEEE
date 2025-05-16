# PDF Query with RAG - FastAPI, Weaviate & Groq

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Weaviate](https://img.shields.io/badge/Weaviate-Client_v4-green.svg)](https://weaviate.io/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3--70b-orange.svg)](https://groq.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-HF-yellow.svg)](https://www.sbert.net/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF%20Processing-red.svg)](https://pymupdf.readthedocs.io/)

This project implements a Retrieval Augmented Generation (RAG) system that allows users to upload PDF documents, process them, and ask questions about their content. It uses FastAPI for the web framework, Weaviate as the vector database, Groq for fast LLM inference (Llama 3 70B), and Sentence Transformers for generating embeddings. The application supports session-based document management, enabling users to query documents within specific contexts.

## Features

* **Session Management**: Create, list, and retrieve sessions to organize document uploads and queries.
* **PDF Upload & Processing**: Upload PDF files which are then processed to extract text.
* **Text Extraction**: Uses PyMuPDF (Fitz) for efficient and accurate text extraction from PDFs.
* **Content Chunking**: Extracted text is divided into manageable chunks for embedding.
* **Vector Embeddings**: Generates embeddings for text chunks using `sentence-transformers/all-MiniLM-L6-v2`.
* **Vector Storage**: Stores text chunks and their corresponding vectors in a Weaviate vector database.
* **Retrieval Augmented Generation (RAG)**:
    * User questions are embedded and used to search for relevant text chunks in Weaviate (semantic search).
    * Retrieved chunks (context) are then passed to a Large Language Model (Groq's Llama 3 70B) along with the user's question to generate an answer.
* **Document Deletion**: Allows deletion of document chunks based on session ID and filename.
* **API Interface**: Exposes all functionalities through a FastAPI-based REST API.
* **Environment Configuration**: Securely manages API keys and URLs using a `.env` file.

## System Architecture (Conceptual)

## Technologies Used

* **Python**: Core programming language.
* **FastAPI**: Modern, fast (high-performance) web framework for building APIs.
* **Weaviate**: Open-source AI-native vector database.
* **Groq**: API for accessing large language models with high inference speed.
* **Sentence Transformers**: Python framework for state-of-the-art sentence, text, and image embeddings.
* **PyMuPDF (Fitz)**: Python binding for MuPDF, a lightweight PDF, XPS, and E-book viewer, renderer, and toolkit.
* **Pydantic**: Data validation and settings management using Python type annotations.
* **Uvicorn**: ASGI server for running FastAPI applications.
* **python-dotenv**: Reads key-value pairs from a `.env` file and sets them as environment variables.
* **UUID**: For generating unique session and document instance IDs.

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <https://github.com/Nuwanga-Wijamuni/RAG_IEEE.git>
    cd <rag-pipeline>
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    Create a `requirements.txt` file with the following content (or add any other specific versions you used):
    ```txt
    fastapi
    uvicorn[standard]
    weaviate-client>=4.0.0 # Ensure you have a version compatible with the script
    groq
    sentence-transformers
    PyMuPDF
    python-dotenv
    pydantic
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory of the project:
    ```env
    WEAVIATE_URL="your_weaviate_cluster_url"
    WEAVIATE_API_KEY="your_weaviate_api_key"
    GROQ_API_KEY="your_groq_api_key"
    ```
    Replace the placeholder values with your actual Weaviate cluster URL, Weaviate API key, and Groq API key.

5.  **Weaviate Schema**:
    The application will attempt to create the necessary "Document" collection in Weaviate on startup if it doesn't exist. The schema includes fields for `session_id`, `document_uuid`, `filename`, `text`, and `chunk_index`.

## Running the Application

Once the setup is complete, you can run the FastAPI application using Uvicorn:

```bash
uvicorn RAGpipline:app --host 127.0.0.1 --port 8000 --reload
