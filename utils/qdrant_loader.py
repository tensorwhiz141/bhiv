"""
Qdrant Document Loader and Pipeline for BHIV Core
Loads sample documents into Qdrant for functional vector search
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import asyncio

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger
from config.settings import QDRANT_CONFIG

logger = get_logger(__name__)

class QdrantDocumentLoader:
    """Loads documents into Qdrant for vector search functionality."""
    
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_CONFIG["url"],
            timeout=60
        )
        self.collection_name = QDRANT_CONFIG["collection_name"]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = 384  # all-MiniLM-L6-v2 output size
        
    def create_collection(self) -> bool:
        """Create Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if self.collection_name in existing_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )
            logger.info(f"Created collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def get_sample_documents(self) -> List[Dict[str, Any]]:
        """Get sample documents for testing the pipeline."""
        return [
            {
                "id": "bhagavad_gita_intro",
                "text": "The Bhagavad Gita is a 700-verse Hindu scripture that is part of the epic Mahabharata. It consists of a conversation between Prince Arjuna and the god Krishna, who serves as his charioteer. The Gita addresses the moral and philosophical dilemma faced by Arjuna, who is reluctant to fight in the Kurukshetra War against his own family members, teachers, and friends.",
                "metadata": {
                    "source": "vedas",
                    "category": "scripture",
                    "language": "english",
                    "subject": "philosophy",
                    "tags": ["gita", "krishna", "arjuna", "dharma"]
                }
            },
            {
                "id": "vedic_education_system",
                "text": "The ancient Vedic education system, known as Gurukula, emphasized holistic development of students. Students lived with their guru (teacher) and learned not just academic subjects but also practical skills, moral values, and spiritual wisdom. The curriculum included the four Vedas, mathematics, astronomy, medicine, and arts.",
                "metadata": {
                    "source": "vedas",
                    "category": "education",
                    "language": "english",
                    "subject": "education",
                    "tags": ["gurukula", "vedic", "education", "guru"]
                }
            },
            {
                "id": "yoga_philosophy",
                "text": "Yoga, derived from the Sanskrit root 'yuj' meaning to unite or join, is both a physical and spiritual practice. The eight limbs of yoga (Ashtanga) as outlined by Patanjali include ethical restraints (yama), observances (niyama), physical postures (asana), breath control (pranayama), withdrawal of senses (pratyahara), concentration (dharana), meditation (dhyana), and ultimate absorption (samadhi).",
                "metadata": {
                    "source": "vedas",
                    "category": "philosophy",
                    "language": "english", 
                    "subject": "yoga",
                    "tags": ["yoga", "patanjali", "ashtanga", "meditation"]
                }
            },
            {
                "id": "ayurveda_principles",
                "text": "Ayurveda, the ancient Indian system of medicine, is based on the concept of three doshas: Vata (air and space), Pitta (fire and water), and Kapha (earth and water). According to Ayurveda, health is maintained when these doshas are in balance, and disease occurs when they are imbalanced. Treatment involves diet, lifestyle changes, herbal medicines, and cleansing procedures.",
                "metadata": {
                    "source": "vedas",
                    "category": "medicine",
                    "language": "english",
                    "subject": "health",
                    "tags": ["ayurveda", "doshas", "health", "medicine"]
                }
            },
            {
                "id": "sanskrit_importance",
                "text": "Sanskrit is considered the language of the gods in Hindu tradition. It is the liturgical language of Hinduism and the scholarly language of ancient and medieval India. Sanskrit has a highly systematic grammar and is known for its precision and clarity. Many modern Indian languages derive their vocabulary from Sanskrit.",
                "metadata": {
                    "source": "vedas",
                    "category": "language",
                    "language": "english",
                    "subject": "linguistics",
                    "tags": ["sanskrit", "language", "hindu", "grammar"]
                }
            },
            {
                "id": "meditation_techniques",
                "text": "Meditation (dhyana) is a fundamental practice in Indian spiritual traditions. Various techniques include focused attention on breath (pranayama), mantras (repetition of sacred sounds), visualization of deities or sacred symbols, and mindfulness of thoughts and sensations. Regular meditation practice leads to mental clarity, emotional stability, and spiritual growth.",
                "metadata": {
                    "source": "vedas",
                    "category": "spirituality",
                    "language": "english",
                    "subject": "meditation",
                    "tags": ["meditation", "dhyana", "mindfulness", "spirituality"]
                }
            },
            {
                "id": "karma_concept",
                "text": "Karma is the universal law of cause and effect in Hindu philosophy. Every action (karma) has consequences that may manifest in this life or future lives. Good actions lead to positive results, while harmful actions lead to suffering. Understanding karma helps individuals take responsibility for their actions and make ethical choices.",
                "metadata": {
                    "source": "vedas",
                    "category": "philosophy",
                    "language": "english",
                    "subject": "ethics",
                    "tags": ["karma", "dharma", "ethics", "philosophy"]
                }
            },
            {
                "id": "bhakti_tradition",
                "text": "Bhakti (devotion) is a spiritual path that emphasizes love and devotion to the divine. Bhakti traditions include singing hymns (bhajans), chanting divine names (kirtan), and performing rituals with deep emotional connection. Great bhakti saints like Mirabai, Tulsidas, and Kabir have enriched this tradition with their devotional poetry and teachings.",
                "metadata": {
                    "source": "vedas", 
                    "category": "spirituality",
                    "language": "english",
                    "subject": "devotion",
                    "tags": ["bhakti", "devotion", "saints", "kirtan"]
                }
            }
        ]

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for documents."""
        embedded_docs = []
        
        for doc in documents:
            try:
                # Generate embedding for the text
                embedding = self.model.encode(doc["text"]).tolist()
                
                embedded_doc = {
                    "id": doc["id"],
                    "vector": embedding,
                    "payload": {
                        "text": doc["text"],
                        "metadata": doc["metadata"]
                    }
                }
                embedded_docs.append(embedded_doc)
                logger.info(f"Generated embedding for document: {doc['id']}")
                
            except Exception as e:
                logger.error(f"Failed to embed document {doc['id']}: {e}")
                
        return embedded_docs

    def upload_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Upload documents to Qdrant collection."""
        try:
            points = []
            for doc in documents:
                point = rest.PointStruct(
                    id=doc["id"],
                    vector=doc["vector"],
                    payload=doc["payload"]
                )
                points.append(point)
            
            # Upload points to collection
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            
            logger.info(f"Uploaded {len(points)} documents to Qdrant. Operation: {operation_info}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload documents: {e}")
            return False

    def test_search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Test search functionality."""
        try:
            # Generate query embedding
            query_vector = self.model.encode(query).tolist()
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload["text"][:200] + "...",
                    "metadata": hit.payload["metadata"]
                }
                results.append(result)
                
            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            points_count = self.client.count(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "vectors_count": points_count.count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"status": "error", "error": str(e)}

    async def initialize_pipeline(self) -> Dict[str, Any]:
        """Initialize the complete Qdrant pipeline."""
        logger.info("Initializing Qdrant pipeline...")
        
        try:
            # Step 1: Create collection
            if not self.create_collection():
                return {"status": "error", "message": "Failed to create collection"}
            
            # Step 2: Check if documents already exist
            collection_info = self.get_collection_info()
            if collection_info.get("vectors_count", 0) > 0:
                logger.info(f"Collection already has {collection_info['vectors_count']} documents")
                return {
                    "status": "success",
                    "message": "Pipeline already initialized",
                    "collection_info": collection_info
                }
            
            # Step 3: Load sample documents
            documents = self.get_sample_documents()
            logger.info(f"Loaded {len(documents)} sample documents")
            
            # Step 4: Generate embeddings
            embedded_docs = self.embed_documents(documents)
            
            # Step 5: Upload to Qdrant
            if not self.upload_documents(embedded_docs):
                return {"status": "error", "message": "Failed to upload documents"}
            
            # Step 6: Test search
            test_results = self.test_search("What is yoga?")
            
            collection_info = self.get_collection_info()
            
            return {
                "status": "success",
                "message": "Qdrant pipeline initialized successfully",
                "documents_loaded": len(embedded_docs),
                "collection_info": collection_info,
                "test_search_results": test_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return {"status": "error", "message": str(e)}

async def main():
    """Main function to initialize the pipeline."""
    loader = QdrantDocumentLoader()
    result = await loader.initialize_pipeline()
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())