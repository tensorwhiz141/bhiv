from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger(__name__)

class VedabaseRetriever:
    def __init__(self, qdrant_url="localhost:6333", collection_name="vedas_knowledge_base"):
        self.client = QdrantClient(qdrant_url, prefer_grpc=False)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = collection_name

    def get_relevant_docs(self, query: str, filters: dict = None, limit: int = 5) -> list:
        """Retrieve relevant documents from Qdrant."""
        try:
            query_vector = self.model.encode(query).tolist()
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filters,
                limit=limit
            )
            results = [hit.payload["text"] for hit in search_result]
            logger.info(f"Retrieved {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            return []