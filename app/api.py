# --------------------------------------------------------------------------
# File: app/api.py
# Description: Defines the FastAPI application and its endpoints.
# --------------------------------------------------------------------------
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, status
from pydantic import BaseModel, Field
from typing import List, Optional
from elasticsearch import ConnectionError

from app.chains import get_full_chain
from app.retriever import get_retriever, get_es_client
from app.ingestion import main as run_ingestion_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="RAG Internship API",
    description="An API for the RAG internship project using Elasticsearch and an open LLM.",
    version="1.0.0",
)

# --- Pydantic Models for API ---

class QueryRequest(BaseModel):
    question: str
    retrieval_mode: str = Field(
        "hybrid", 
        description="Retrieval mode: 'hybrid' or 'elser_only'."
    )

class Citation(BaseModel):
    filename: str
    snippet: str
    drive_url: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]

class HealthResponse(BaseModel):
    api_status: str
    elasticsearch_status: str

# --- API Endpoints ---

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Accepts a user question and returns a grounded answer with citations.
    """
    logging.info(f"Received query: '{request.question}' with mode: '{request.retrieval_mode}'")
    
    try:
        # 1. Get the appropriate retriever based on the requested mode
        retriever = get_retriever(mode=request.retrieval_mode)

        # 2. Get the full RAG chain, configured with the chosen retriever
        chain = get_full_chain(retriever=retriever)

        # 3. Invoke the chain with the user's question
        logging.info("Invoking RAG chain...")
        result = chain.invoke({"question": request.question})
        logging.info("RAG chain invocation complete.")

        # 4. Format the citations from the retrieved context
        citations = []
        if result.get("context"):
            for doc in result["context"]:
                citations.append(
                    Citation(
                        filename=doc.metadata.get("filename", "Unknown"),
                        snippet=doc.page_content,
                        drive_url=doc.metadata.get("drive_url"),
                    )
                )

        return QueryResponse(answer=result["answer"], citations=citations)

    except Exception as e:
        # This is the new, more detailed error logging.
        # It will print the full traceback to your console.
        logging.error(f"An error occurred during query processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred: {str(e)}"
        )

@app.post("/ingest")
def ingest_documents(background_tasks: BackgroundTasks):
    """
    Triggers the ingestion process from configured sources (local folder and/or Google Drive)
    to re-index the data in Elasticsearch. This runs as a background task.
    """
    logging.info("Ingestion endpoint triggered.")
    background_tasks.add_task(run_ingestion_pipeline)
    return {"message": "Ingestion process started in the background. Check server logs for progress."}


@app.get("/healthz", response_model=HealthResponse)
def health_check(response: Response):
    """
    Checks the status of the API and its connection to Elasticsearch.
    """
    es_status = "ok"
    try:
        es_client = get_es_client()
        if not es_client.ping():
            raise ConnectionError("Elasticsearch ping failed")
    except ConnectionError as e:
        logging.error(f"Elasticsearch connection health check failed: {e}")
        es_status = "error"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(api_status="ok", elasticsearch_status=es_status)

