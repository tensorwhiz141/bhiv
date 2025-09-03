# # ```python
# from qdrant_client import QdrantClient
# from config.settings import QDRANT_CONFIG, TIMEOUT_CONFIG
# from utils.logger import get_logger
# import asyncio
# from typing import Dict, List, Any, Optional

# logger = get_logger(__name__)

# class KnowledgeAgent:
#     def __init__(self):
#         self.qdrant_client = QdrantClient(
#             url=QDRANT_CONFIG["url"],
#             timeout=TIMEOUT_CONFIG.get("qdrant_query_timeout", 60)
#         )
#         self.collection_name = QDRANT_CONFIG["collection_name"]

#     async def query(self, query: str, task_id: str, filters: Dict[str, Any], tags: List[str]) -> Dict[str, Any]:
#         logger.info(f"[KnowledgeAgent] Task {task_id}: Querying Qdrant with query: {query}, filters: {filters}, tags: {tags}")
#         try:
#             # Placeholder for Qdrant query logic
#             search_results = self.qdrant_client.search(
#                 collection_name=self.collection_name,
#                 query_vector=[0.1] * QDRANT_CONFIG["vector_size"],  # Dummy vector for testing
#                 query_filter=filters,
#                 limit=5
#             )
#             results = [
#                 {
#                     "text": point.payload.get("text", ""),
#                     "metadata": point.payload.get("metadata", {})
#                 }
#                 for point in search_results
#             ]
#             return {
#                 "query_id": task_id,
#                 "query": query,
#                 "response": results,
#                 "status": 200,
#                 "metadata": {"source": "vedas_knowledge_base"}
#             }
#         except Exception as e:
#             logger.error(f"[KnowledgeAgent] Task {task_id}: Query failed: {str(e)}")
#             return {
#                 "query_id": task_id,
#                 "query": query,
#                 "response": [],
#                 "status": 500,
#                 "metadata": {"error": str(e)}
#             }

#     async def process_task(self, task_id: str, input_data: str, input_type: str, pdf_path: Optional[str], tags: List[str]) -> Dict[str, Any]:
#         logger.info(f"[KnowledgeAgent] Task {task_id}: Processing task with input: {input_data}, type: {input_type}, tags: {tags}")
#         return await self.query(query=input_data, task_id=task_id, filters={}, tags=tags)
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from config.settings import QDRANT_CONFIG, TIMEOUT_CONFIG
from utils.logger import get_logger
from sentence_transformers import SentenceTransformer
import asyncio
from typing import Dict, List, Any, Optional

logger = get_logger(__name__)

class KnowledgeAgent:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=QDRANT_CONFIG["url"],
            timeout=TIMEOUT_CONFIG.get("qdrant_query_timeout", 60)
        )
        self.collection_name = QDRANT_CONFIG["collection_name"]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Same model as loader
        logger.info(f"KnowledgeAgent initialized with collection: {self.collection_name}")

    async def query(self, query: str, task_id: str, filters: Dict[str, Any], tags: List[str]) -> Dict[str, Any]:
        logger.info(f"[KnowledgeAgent] Task {task_id}: Querying Qdrant with query: {query}, filters: {filters}, tags: {tags}")
        try:
            # Generate query vector using the embedding model
            query_vector = self.model.encode(query).tolist()
            
            # Convert filters to Qdrant Filter object if provided
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(rest.FieldCondition(
                        key=f"metadata.{key}",
                        match=rest.MatchValue(value=value)
                    ))
                if conditions:
                    qdrant_filter = rest.Filter(must=conditions)
            
            # Perform search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=5,
                with_payload=True
            )
            
            # Format results
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "text": hit.payload.get("text", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "score": float(hit.score),
                    "relevance": "high" if hit.score > 0.8 else "medium" if hit.score > 0.6 else "low"
                }
                results.append(result)
            
            logger.info(f"[KnowledgeAgent] Task {task_id}: Found {len(results)} relevant documents")
            
            return {
                "query_id": task_id,
                "query": query,
                "response": results,
                "status": 200,
                "metadata": {
                    "source": "vedas_knowledge_base",
                    "search_method": "vector_similarity",
                    "total_results": len(results),
                    "collection": self.collection_name
                }
            }
            
        except Exception as e:
            logger.error(f"[KnowledgeAgent] Task {task_id}: Query failed: {str(e)}")
            return {
                "query_id": task_id,
                "query": query,
                "response": [],
                "status": 500,
                "metadata": {"error": str(e)}
            }

    async def process_task(self, task_id: str, input_data: str, input_type: str, pdf_path: Optional[str], tags: List[str]) -> Dict[str, Any]:
        logger.info(f"[KnowledgeAgent] Task {task_id}: Processing task with input: {input_data}, type: {input_type}, tags: {tags}")
        return await self.query(query=input_data, task_id=task_id, filters={}, tags=tags)