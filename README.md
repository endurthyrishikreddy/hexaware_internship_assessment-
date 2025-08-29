# RAG System with Elasticsearch and an Open LLM

This repository contains the code for a **simplified Retrieval-Augmented Generation (RAG) system** built as an internship project. The system uses **Elasticsearch** for advanced hybrid retrieval and an **open-source LLM (via Ollama)** for answer generation.

---

## 📌 Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Project Structure](#project-structure)  
- [Setup and Installation](#setup-and-installation)  
- [How to Run](#how-to-run)  
- [API Endpoints](#api-endpoints)  
- [Demo](#demo)  

---

## 📝 Overview
This project implements a complete **RAG pipeline** that can answer questions based on a collection of **PDF documents**.  

- Documents are ingested from a **local folder** and/or a **Google Drive folder**.  
- Indexed into **Elasticsearch** with a **hybrid retrieval strategy**.  
- Retrieved context is used by an **open-source LLM (via Ollama)** to generate grounded answers.  
- Exposed via **FastAPI backend** and a **Streamlit web interface**.  

---

## ✨ Features
- **Dual Ingestion**:  
  - Local `data/` folder  
  - Google Drive folder  

- **Advanced Hybrid Retrieval**: Combines  
  - **ELSER** (Elastic sparse model for semantic search)  
  - **Dense Vectors** (via Sentence-Transformers)  
  - **BM25** (keyword search)  

- **Open-Source LLM via Ollama**:  
  - Supports **Llama 3, Phi-3**, or any available Ollama model  
  - No dependency on paid APIs  

- **Grounded Answers with Citations**:  
  - Answers include **source filename** + **snippet citation**  

- **API + UI**:  
  - **FastAPI backend**  
  - **Streamlit frontend**  

- **Safety Guardrails**:  
  - Prompt-based guardrails to **restrict off-topic answers**  

---

## 🏗️ Architecture
The system has **two pipelines**:

### 🔹 Indexing Pipeline (Offline)
1. Load documents from **local / Google Drive**  
2. Split into smaller chunks  
3. Index chunks for **BM25 search**  
4. Generate **dense vector embeddings**  
5. Create **ELSER sparse embeddings**  
6. Store all data in a single **Elasticsearch index**  

### 🔹 Retrieval & Generation Pipeline (Online)
1. User submits a question  
2. `HybridRetriever` sends query to Elasticsearch  
3. Elasticsearch performs **RRF (Reciprocal Rank Fusion)** on BM25, Dense, and ELSER  
4. Retrieve top-ranked document chunks  
5. Pass **context + question + system prompt** to Ollama LLM  
6. LLM generates final grounded answer with citations  

---

## 📂 Project Structure
![Project Structure](https://github.com/user-attachments/assets/3064cecc-d70b-4fb5-853b-a45853289e02)

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python **3.11+**  
- **Docker** & **Docker Compose**  
- **Git**  
- **Google Cloud Project** with Drive API enabled  
- **Tesseract OCR engine**  

### 2. Clone the Repository
```
git clone https://github.com/endurthyrishikreddy/hexaware.git
cd hexaware
```


### 3. Set Up the Python Environment
- It is highly recommended to use a virtual environment.

```
conda create --name rag_env python=3.11 -y
conda activate rag_env
pip install -r requirements.txt
```


### 4. Configure Environment Variables
- Rename .env.example to .env.

 - Fill in the required values for your Elasticsearch URL, API Key, and Google Drive Folder ID.

# .env
```
ELASTIC_URL=http://localhost:9200
ELASTIC_API_KEY=your_api_key_here
INDEX_NAME=rag-internship-demo
GOOGLE_DRIVE_FOLDER_ID=your_gdrive_folder_id_here
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=phi3
OLLAMA_BASE_URL=http://localhost:11434
```

### 5. Google Drive Credentials
- Follow the Google Drive API Python quickstart to get your credentials.json file.

- Place the credentials.json file in the root directory.

- The first time you run the ingestion script, you will be prompted to authorize the application in your browser. This will create a google_token.json file.

6. Start Docker Services
Start the Elasticsearch and Ollama containers in detached mode.

```docker-compose up -d --build```

Deploy ELSER: You must deploy the ELSER model in Elasticsearch. For serverless, this is typically available by default. For managed clusters, you may need to deploy it via the Kibana UI.

Pull an LLM: Download the model specified in your .env file into the Ollama container.

# Find the container name first with 'docker ps'
```docker exec -it <your_ollama_container_name> ollama pull phi3```

How to Run
# Step 1: Ingest Data
Run the ingestion script from the project's root directory. This will process all PDFs from the data/ folder and your configured Google Drive folder.

```python -m app.ingestion```

# Step 2: Start the FastAPI Backend
In a new terminal, start the API server from the root directory.

```uvicorn app.api:app --reload --host 0.0.0.0 --port 8000```

The API will be available at http://localhost:8000. You can access the auto-generated documentation at http://localhost:8000/docs.

# Step 3: Launch the Streamlit UI
In a third terminal, launch the user interface.

```streamlit run app/ui.py```

The UI will open in your browser, typically at http://localhost:8501.

# API Endpoints
POST /query

Accepts a JSON body with a question and an optional retrieval_mode ('hybrid' or 'elser_only').

Returns a JSON object with the answer and a list of citations.
<img width="1076" height="974" alt="image" src="https://github.com/user-attachments/assets/63e89c6a-ffa3-4154-b6b9-663a255fb162" />


POST /ingest

Triggers the data ingestion pipeline as a background task.
<img width="1435" height="843" alt="image" src="https://github.com/user-attachments/assets/682b023f-d2e7-4924-9256-7d60c868cb1c" />


Returns an immediate confirmation message.

GET /healthz

A health check endpoint that verifies API and Elasticsearch connectivity.
<img width="1417" height="997" alt="image" src="https://github.com/user-attachments/assets/0cc9ef70-d197-4ac4-a8e7-0d1c85b56a72" />


# Demo
Screenshots
(Placeholder: Insert screenshots of your Streamlit UI here)

<img width="1895" height="991" alt="image" src="https://github.com/user-attachments/assets/b3b84d9d-cf8c-4fac-ad93-eb15faf756da" />

