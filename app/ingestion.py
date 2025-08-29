# --------------------------------------------------------------------------
# File: app/ingestion.py (Updated for Hybrid Local and Google Drive Ingestion)
# Description: Handles loading, processing, and indexing of documents from a local directory and Google Drive.
# --------------------------------------------------------------------------
import logging
import os
from elasticsearch import Elasticsearch
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_google_community import GoogleDriveLoader

from app.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for Local Ingestion ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOCAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "credentials.json")
TOKEN_PATH = os.path.join(PROJECT_ROOT, "token.json")


def get_es_client():
    """Gets an Elasticsearch client instance, handling different auth methods."""
    # Case 1: Connection using Elastic Cloud ID and API Key
    if settings.elastic_cloud_id and settings.elastic_api_key:
        logging.info("Connecting to Elastic Cloud with Cloud ID.")
        return Elasticsearch(
            cloud_id=settings.elastic_cloud_id,
            api_key=settings.elastic_api_key
        )
    # Case 2: Connection using URL and API Key
    elif settings.elastic_url and settings.elastic_api_key:
        logging.info(f"Connecting to Elastic URL with API Key: {settings.elastic_url}")
        return Elasticsearch(
            settings.elastic_url,
            api_key=settings.elastic_api_key
        )
    # Case 3: Connection using only URL (e.g., local, unauthenticated)
    elif settings.elastic_url:
        logging.info(f"Connecting to Elastic URL (no auth): {settings.elastic_url}")
        return Elasticsearch(settings.elastic_url)
    # Error case if no settings are provided
    else:
        raise ValueError("No Elasticsearch connection settings found. Please configure ELASTIC_URL or ELASTIC_CLOUD_ID.")


def load_docs_from_local_folder():
    """
    Loads PDF documents from a local folder with robust error handling for each file.
    Switched to PyPDFLoader to avoid unstructured dependency issues.
    """
    if not os.path.exists(LOCAL_DATA_PATH):
        logging.warning(f"Local data directory not found at: {LOCAL_DATA_PATH}. Skipping local ingestion.")
        return []
        
    logging.info(f"Loading documents from local directory: {LOCAL_DATA_PATH}")
    
    loaded_docs = []
    pdf_files = [f for f in os.listdir(LOCAL_DATA_PATH) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        file_path = os.path.join(LOCAL_DATA_PATH, pdf_file)
        try:
            # Use PyPDFLoader for more stability and fewer dependencies
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            loaded_docs.extend(docs)
            logging.info(f"Successfully loaded {pdf_file}")
        except Exception as e:
            logging.error(f"Failed to load or process {pdf_file}. Error: {e}", exc_info=True)
            # Continue to the next file
            continue
            
    logging.info(f"Loaded a total of {len(loaded_docs)} documents from the local folder.")
    return loaded_docs


def load_docs_from_gdrive():
    """Loads PDF documents from the specified Google Drive folder."""
    if not os.path.exists(CREDENTIALS_PATH):
        logging.warning(f"Google Drive credentials not found at {CREDENTIALS_PATH}. Skipping Google Drive ingestion.")
        return []

    logging.info(f"Loading documents from Google Drive folder: {settings.google_drive_folder_id}")
    try:
        loader = GoogleDriveLoader(
            folder_id=settings.google_drive_folder_id,
            credentials_path=CREDENTIALS_PATH,
            token_path=TOKEN_PATH,
            file_loader_cls=UnstructuredPDFLoader,
            file_loader_kwargs={"strategy": "ocr_only"},
            recursive=True,
            file_types=["pdf"],
        )
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents from Google Drive.")
        return docs
    except Exception as e:
        logging.error(f"Failed to load documents from Google Drive. Error: {e}", exc_info=True)
        return []


def chunk_documents(docs):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    logging.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    
    # Add chunk_id and standardize metadata for both sources
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        if 'source' in chunk.metadata:
            chunk.metadata['filename'] = os.path.basename(chunk.metadata['source'])
        # Create a Google Drive URL only if file_id is present
        if 'file_id' in chunk.metadata:
             chunk.metadata['drive_url'] = f"https://docs.google.com/document/d/{chunk.metadata['file_id']}/"

    return chunks

def create_index_if_not_exists(client, index_name):
    """Creates an Elasticsearch index with specific mappings if it doesn't exist."""
    if client.indices.exists(index=index_name):
        logging.info(f"Index '{index_name}' already exists. Skipping creation.")
        return

    logging.info(f"Creating index '{index_name}' with custom mappings.")
    
    # Define the mapping for hybrid search
    mappings = {
        "properties": {
            "text": {"type": "text"},
            "vector": {
                "type": "dense_vector",
                "dims": 384,  # Corresponds to all-MiniLM-L6-v2
                "index": True,
                "similarity": "cosine"
            },
            "text_expansion": {
                "type": "rank_features" 
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "filename": {"type": "keyword"},
                    "drive_url": {"type": "keyword"},
                    "chunk_id": {"type": "integer"},
                    "page": {"type": "integer"}
                }
            }
        }
    }
    
    client.indices.create(index=index_name, mappings=mappings)
    logging.info("Index created successfully.")


def main():
    """Main ingestion pipeline."""
    logging.info("Starting ingestion process...")
    
    # 1. Load documents
    local_docs = load_docs_from_local_folder()
    gdrive_docs = load_docs_from_gdrive()
    all_docs = local_docs + gdrive_docs
    
    if not all_docs:
        logging.warning("No documents found. Exiting ingestion.")
        return

    logging.info(f"Total documents loaded: {len(all_docs)}")

    # 2. Chunk documents
    chunks = chunk_documents(all_docs)

    # 3. Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    # 4. Get ES client and create index with mapping
    logging.info("Connecting to Elasticsearch...")
    es_client = get_es_client()
    create_index_if_not_exists(es_client, settings.index_name)
    
    # 5. Index into Elasticsearch
    logging.info(f"Indexing chunks into Elasticsearch index: {settings.index_name}")

    db = ElasticsearchStore.from_documents(
        chunks,
        embeddings,
        es_connection=es_client,
        index_name=settings.index_name,
        # The strategy is no longer needed here as the index and mappings are pre-defined
    )
    db.client.indices.refresh(index=settings.index_name)
    logging.info("Ingestion process completed successfully.")

if __name__ == "__main__":
    main()

