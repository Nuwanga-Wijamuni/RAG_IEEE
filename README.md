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
