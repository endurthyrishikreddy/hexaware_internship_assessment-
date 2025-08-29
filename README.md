RAG System with Elasticsearch and an Open LLM
This repository contains the code for a simplified Retrieval-Augmented Generation (RAG) system built as an internship project. The system uses Elasticsearch for advanced hybrid retrieval and an open-source LLM (via Ollama) for answer generation.

A high-level diagram of the RAG pipeline.

Table of Contents
Overview

Features

Architecture

Project Structure

Setup and Installation

How to Run

API Endpoints

Demo

Overview
This project implements a complete RAG pipeline that can answer questions based on a collection of PDF documents. It ingests documents from a local folder and/or a Google Drive folder, indexes them into Elasticsearch, and uses a hybrid retrieval strategy to find the most relevant context for an open-source LLM to synthesize an answer. The system is accessible via a FastAPI backend and a user-friendly Streamlit web interface.

Features
Dual Ingestion: Ingests PDF documents from both a local data/ folder and a specified Google Drive folder.

Advanced Hybrid Retrieval: Utilizes Elasticsearch to perform a hybrid search combining:

ELSER: Elastic's powerful sparse vector model for semantic search.

Dense Vectors: A sentence-transformer model for dense vector similarity search.

BM25: Traditional keyword-based search for exact matches.

Open-Source LLM: Uses a locally hosted open-source model (e.g., Llama 3, Phi-3) via Ollama for answer generation, ensuring no dependency on paid APIs.

Grounded Answers with Citations: Responses are grounded in the retrieved documents and include citations (source filename and snippet).

API & UI: Provides a robust FastAPI backend and an interactive Streamlit UI for Q&A.

Safety Guardrails: Implements prompt-based guardrails to ensure answers are based only on the provided context and refuse off-topic questions.

Architecture
The system is divided into two main pipelines:

Indexing Pipeline (Offline):

Documents are loaded from local or Google Drive sources.

They are split into smaller, manageable chunks.

Each chunk's text is indexed for BM25 search.

A dense vector embedding is generated and stored.

ELSER processes the text to create a sparse vector embedding.

All data and embeddings are stored in a single Elasticsearch index.

Retrieval & Generation Pipeline (Online):

A user asks a question through the UI or API.

The question is sent to the custom HybridRetriever.

Elasticsearch performs a hybrid search using RRF (Reciprocal Rank Fusion) to combine and rank the results from BM25, dense, and ELSER searches.

The top-ranked document chunks are retrieved.

The context, user question, and a detailed system prompt are passed to the Ollama LLM.

The LLM generates a final answer, which is returned to the user along with the source citations.

Project Structure
rag_gemini/
├── app/
│   ├── __init__.py
│   ├── api.py           # FastAPI application
│   ├── chains.py        # LangChain (LCEL) chain logic
│   ├── config.py        # Pydantic settings management
│   ├── ingestion.py     # Data ingestion and indexing script
│   ├── retriever.py     # Custom Elasticsearch hybrid retriever
│   └── ui.py            # Streamlit user interface
├── data/                  # Local folder for PDF documents
├── .env                   # Environment variables (credentials, etc.)
├── .env.example           # Example environment file
├── .gitignore             # Files to ignore in git
├── credentials.json       # Google Drive API credentials
├── docker-compose.yml     # Docker services for Elasticsearch and Ollama
├── README.md              # This file
└── requirements.txt       # Python dependencies

Setup and Installation
Follow these steps to set up the project locally.

1. Prerequisites
Python 3.11+

Docker and Docker Compose

Git

Google Cloud Project with Drive API enabled (for G-Drive ingestion)

Tesseract OCR engine (for PDF processing)

2. Clone the Repository
git clone [https://github.com/endurthyrishikreddy/hexaware.git](https://github.com/endurthyrishikreddy/hexaware.git)
cd hexaware

3. Set Up the Python Environment
It is highly recommended to use a virtual environment.

conda create --name rag_env python=3.11 -y
conda activate rag_env
pip install -r requirements.txt

4. Configure Environment Variables
Rename .env.example to .env.

Fill in the required values for your Elasticsearch URL, API Key, and Google Drive Folder ID.

# .env
ELASTIC_URL=http://localhost:9200
ELASTIC_API_KEY=your_api_key_here
INDEX_NAME=rag-internship-demo
GOOGLE_DRIVE_FOLDER_ID=your_gdrive_folder_id_here
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=phi3
OLLAMA_BASE_URL=http://localhost:11434

5. Google Drive Credentials
Follow the Google Drive API Python quickstart to get your credentials.json file.

Place the credentials.json file in the root directory.

The first time you run the ingestion script, you will be prompted to authorize the application in your browser. This will create a google_token.json file.

6. Start Docker Services
Start the Elasticsearch and Ollama containers in detached mode.

docker-compose up -d --build

Deploy ELSER: You must deploy the ELSER model in Elasticsearch. For serverless, this is typically available by default. For managed clusters, you may need to deploy it via the Kibana UI.

Pull an LLM: Download the model specified in your .env file into the Ollama container.

# Find the container name first with 'docker ps'
docker exec -it <your_ollama_container_name> ollama pull phi3

How to Run
Step 1: Ingest Data
Run the ingestion script from the project's root directory. This will process all PDFs from the data/ folder and your configured Google Drive folder.

python -m app.ingestion

Step 2: Start the FastAPI Backend
In a new terminal, start the API server from the root directory.

uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

The API will be available at http://localhost:8000. You can access the auto-generated documentation at http://localhost:8000/docs.

Step 3: Launch the Streamlit UI
In a third terminal, launch the user interface.

streamlit run app/ui.py

The UI will open in your browser, typically at http://localhost:8501.

API Endpoints
POST /query

Accepts a JSON body with a question and an optional retrieval_mode ('hybrid' or 'elser_only').

Returns a JSON object with the answer and a list of citations.

POST /ingest

Triggers the data ingestion pipeline as a background task.

Returns an immediate confirmation message.

GET /healthz

A health check endpoint that verifies API and Elasticsearch connectivity.

Demo
Screenshots
(Placeholder: Insert screenshots of your Streamlit UI here)

Caption for the first screenshot.

Caption for the second screenshot.

Demo Video
(Placeholder: Insert a link to your 5-minute demo video here)

Link to Demo Video
