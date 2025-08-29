# --------------------------------------------------------------------------
# File: app/retriever.py
# Description: Defines the hybrid retriever logic for Elasticsearch.
# --------------------------------------------------------------------------
import logging
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_es_client():
    """Gets an Elasticsearch client instance."""
    if settings.elastic_cloud_id:
        logging.info(f"Connecting to Elastic Cloud: {settings.elastic_cloud_id}")
        return Elasticsearch(
            cloud_id=settings.elastic_cloud_id,
            api_key=settings.elastic_api_key
        )
    elif settings.elastic_url and settings.elastic_api_key:
        logging.info(f"Connecting to Elastic URL with API Key: {settings.elastic_url}")
        return Elasticsearch(
            hosts=[settings.elastic_url],
            api_key=settings.elastic_api_key
        )
    logging.info(f"Connecting to Elastic URL (unauthenticated): {settings.elastic_url}")
    return Elasticsearch(hosts=[settings.elastic_url])

class HybridRetriever(BaseRetriever):
    """
    Custom retriever that combines dense, sparse (ELSER), and keyword (BM25) search
    using Reciprocal Rank Fusion (RRF) for ranking.
    """
    client: Elasticsearch
    index_name: str
    embedding_model: HuggingFaceEmbeddings
    k: int = 5
    mode: str = "hybrid"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Executes the hybrid retrieval logic.
        """
        logging.info(f"Executing retrieval with mode: '{self.mode}'")
        
        # ELSER Sparse Vector Search query
        elser_query = {
            "text_expansion": {
                "elser_embedding_field": {
                    "model_id": "elser",
                    "model_text": query,
                }
            }
        }
        
        try:
            if self.mode == "elser_only":
                # Execute ELSER-only search
                response = self.client.search(
                    index=self.index_name,
                    query=elser_query,
                    size=self.k
                )
            else:
                # Execute Hybrid Search
                # Dense Vector Search (HNSW) parameters
                knn_query = {
                    "field": "vector_field",
                    "query_vector": self.embedding_model.embed_query(query),
                    "k": self.k,
                    "num_candidates": 50,
                }

                # BM25 Keyword Search query
                bm25_query = {"match": {"text_content": {"query": query}}}

                # RRF Rank configuration
                rank_config = {"rrf": {"rank_constant": 20, "window_size": 100}}

                # Pass search components as keyword arguments instead of a single 'body' dict.
                # This is a more robust way to build the query.
                response = self.client.search(
                    index=self.index_name,
                    knn=knn_query,
                    sub_searches=[
                        {"query": bm25_query},
                        {"query": elser_query},
                    ],
                    rank=rank_config,
                    size=self.k,
                )
            
            docs = []
            for hit in response["hits"]["hits"]:
                docs.append(
                    Document(
                        page_content=hit["_source"]["text_content"],
                        metadata=hit["_source"]["metadata"],
                    )
                )
            return docs

        except Exception as e:
            logging.error(f"Error during Elasticsearch retrieval: {e}", exc_info=True)
            return []

def get_retriever(mode: str = "hybrid", k: int = 5):
    """
    Factory function to create an instance of the HybridRetriever.
    """
    es_client = get_es_client()
    embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    
    return HybridRetriever(
        client=es_client,
        index_name=settings.index_name,
        embedding_model=embedding_model,
        k=k,
        mode=mode,
    )

