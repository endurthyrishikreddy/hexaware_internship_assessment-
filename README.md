RAG System with Elasticsearch and Open LLM
This project is a simplified, end-to-end Retrieval-Augmented Generation (RAG) system built for the internship assignment. It uses Elasticsearch for hybrid retrieval (ELSER, dense vectors, BM25) and an open-source LLM for answer generation.

Architecture Overview
The system follows a standard RAG pipeline:

Ingestion: PDF documents are loaded from a specified Google Drive folder, split into chunks, and indexed into an Elasticsearch index. Each chunk is indexed with its raw text (for BM25), a dense vector embedding, and an ELSER sparse vector.

Retrieval: When a user asks a question, the system queries Elasticsearch using a hybrid search approach, combining BM25, dense vector search, and ELSER with Reciprocal Rank Fusion (RRF) to find the most relevant document chunks.

Generation: The retrieved chunks are passed as context to an open-source LLM (like Llama 3), which generates a grounded answer based only on the provided information.

Interface: The system is accessible via a FastAPI backend and a user-friendly Streamlit web UI.

Features
Document Source: Ingests PDFs directly from a Google Drive folder.

Hybrid Search: Leverages Elasticsearch for three retrieval methods:

BM25: Keyword-based search.

Dense Vector: Semantic search using sentence-transformer embeddings.

ELSER: Elastic's sparse vector model for advanced semantic search.

Reciprocal Rank Fusion (RRF): Intelligently combines results from all three search methods.

Open-Source LLM: Uses a local LLM (e.g., Llama 3 via Ollama) for generation, avoiding paid APIs.

Grounded Answers & Citations: Provides answers with citations (document source, link, and text snippet) to ensure traceability.

Built-in Guardrails: Includes basic safety checks to prevent harmful or off-topic responses and ensures answers are derived from the source documents.

Dual Interface: Offers both a REST API (FastAPI) and an interactive web UI (Streamlit).

Prerequisites
Python 3.9+

Docker and Docker Compose

Access to a Google Drive folder with some PDF files.

Setup Instructions
1. Clone the Repository

git clone <your-repo-url>
cd rag-elastic-internship

2. Set up Google Drive API Credentials

To allow the application to access your Google Drive, you need to enable the Google Drive API and get credentials.

Go to the Google Cloud Console.

Create a new project.

Enable the Google Drive API.

Create credentials for a Desktop app.

Download the credentials.json file and place it in the root of this project directory.

The first time you run the ingestion script, you will be prompted to authorize access. A token.json file will be created automatically to store your authorization.

3. Create the .env File

Create a file named .env in the root directory and populate it with your configuration.

# Elasticsearch Configuration
ELASTIC_URL="http://localhost:9200"
# ELASTIC_API_KEY="Your_API_Key_If_Using_Elastic_Cloud" # Uncomment if using Elastic Cloud
ELASTIC_INDEX="pdf_rag_index"

# Google Drive Configuration
GDRIVE_FOLDER_ID="your_google_drive_folder_id"

# Embedding and LLM Configuration
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL="http://localhost:11434"
LLM_MODEL_NAME="llama3"

# Retrieval Configuration
TOP_K=5

4. Set up Elasticsearch and Ollama with Docker

We'll use Docker Compose to run Elasticsearch and Ollama.

Create a docker-compose.yml file in the project root:

version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.1
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ELASTIC_PASSWORD=
    ports:
      - "9200:9200"
    volumes:
      - esdata:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -q '\"status\":\"green\"'"]
      interval: 10s
      timeout: 10s
      retries: 5

  ollama:
    image: ollama/ollama
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollamadata:/root/.ollama

volumes:
  esdata:
    driver: local
  ollamadata:
    driver: local

Start the services:

docker-compose up -d

Important: You need to pull the ELSER model into Elasticsearch and the LLM model into Ollama.

# Pull the ELSER model (wait for Elasticsearch to be healthy first)
docker exec -it elasticsearch bin/elasticsearch-ml-model-management-tool download --model-id=.elser_model_2

# Pull the LLM model
docker exec -it ollama ollama pull llama3

5. Install Python Dependencies

pip install -r requirements.txt

How to Run the System
Step 1: Ingest Documents

Run the ingestion script to load, chunk, and index the PDFs from your Google Drive folder into Elasticsearch.

python ingestion.py

Step 2: Start the FastAPI Server

This runs the backend API.

uvicorn api:app --reload

You can access the API docs at http://127.0.0.1:8000/docs.

Step 3: Run the Streamlit UI

This starts the web interface.

streamlit run ui.py

Open your browser and go to http://localhost:8501.

API Endpoints
POST /ingest

Triggers the ingestion process from Google Drive.

Body: {}

POST /query

Asks a question to the RAG system.

Body: {"question": "Your question here?", "mode": "hybrid"}

mode can be hybrid or elser_only.

GET /healthz

A simple health check endpoint.