# --------------------------------------------------------------------------
# File: app/config.py
# Description: Manages application settings using Pydantic for validation.
# --------------------------------------------------------------------------
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Pydantic settings model to load environment variables.
    """
    # Elasticsearch settings
    elastic_url: str = "http://localhost:9200"
    elastic_cloud_id: str | None = None
    elastic_api_key: str | None = None
    index_name: str = "rag-internship-demo"

    # Google Drive settings
    google_drive_folder_id: str = "YOUR_GOOGLE_DRIVE_FOLDER_ID"

    # Ingestion settings
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Model settings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434" # Ollama server URL

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# Instantiate settings
settings = Settings()
