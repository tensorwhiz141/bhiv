import os
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from config.settings import MODEL_CONFIG

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelProvider:
    def __init__(self, model_config: Dict[str, Any], endpoint: str):
        self.model = None
        self.model_name = model_config.get("model_name", "llama3-8b-8192")  # Updated model
        self.api_key = os.getenv("GROQ_API_KEY")
        self.backup_api_key = os.getenv("GROQ_API_KEY_BACKUP")
        self.endpoint = endpoint
        self.initialize_model()

    def initialize_model(self):
        try:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not found")
            self.model = ChatGroq(
                model=self.model_name,
                api_key=self.api_key
            )
            logger.info(f"Initialized Groq model: {self.model_name} for {self.endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize model for {self.endpoint} with primary key: {e}")
            if self.backup_api_key:
                try:
                    logger.info(f"Trying backup API key for {self.endpoint}")
                    self.model = ChatGroq(
                        model=self.model_name,
                        api_key=self.backup_api_key
                    )
                    logger.info(f"Initialized Groq model with backup key: {self.model_name}")
                except Exception as e:
                    logger.error(f"Backup key failed for {self.endpoint}: {e}")
                    self.model = None
            else:
                self.model = None

    def generate_response(self, prompt: str, fallback: str) -> tuple[str, int]:
        if not self.model:
            logger.warning(f"No model initialized for {self.endpoint}. Using fallback response.")
            return fallback, 500
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            return response.content.strip(), 200
        except Exception as e:
            logger.error(f"Model generation error for {self.endpoint}: {e}")
            return fallback, 500

class SimpleOrchestrationEngine:
    def __init__(self):
        self.vector_stores = {}
        self.embedding_model = None
        self.model_providers = {
            "vedas": ModelProvider(MODEL_CONFIG["vedas_agent"], "ask-vedas"),
            "edumentor": ModelProvider(MODEL_CONFIG["edumentor_agent"], "edumentor"),
            "wellness": ModelProvider(MODEL_CONFIG["wellness_agent"], "wellness")
        }
        self.initialize_vector_stores()
        
    def initialize_vector_stores(self):
        logger.info("Initializing embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store_dir = Path("vector_stores")
        store_names = ['vedas_index', 'wellness_index', 'educational_index', 'unified_index']
        for store_name in store_names:
            store_path = vector_store_dir / store_name
            if store_path.exists():
                try:
                    store = FAISS.load_local(
                        str(store_path), 
                        self.embedding_model, 
                        allow_dangerous_deserialization=True
                    )
                    self.vector_stores[store_name.replace('_index', '')] = store
                    logger.info(f"Loaded vector store: {store_name}")
                except Exception as e:
                    logger.error(f"Failed to load vector store {store_name}: {e}")
        logger.info(f"Initialized with {len(self.vector_stores)} vector stores")
    
    def generate_response(self, prompt: str, fallback: str, endpoint: str) -> tuple[str, int]:
        return self.model_providers[endpoint].generate_response(prompt, fallback)
    
    def search_documents(self, query: str, store_type: str = "unified") -> list:
        if store_type in self.vector_stores:
            try:
                retriever = self.vector_stores[store_type].as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)
                return [{"text": doc.page_content[:500], "source": doc.metadata.get("source", "unknown")} for doc in docs]
            except Exception as e:
                logger.error(f"Vector search error: {e}")
        return []

engine = SimpleOrchestrationEngine()

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "anonymous"

class SimpleResponse(BaseModel):
    query_id: str
    query: str
    response: str
    sources: list
    timestamp: str
    endpoint: str
    status: int  # Added status field

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Simple Orchestration API...")
    engine.initialize_vector_stores()
    logger.info("Simple Orchestration API ready!")
    yield
    logger.info("Shutting down Simple Orchestration API...")

app = FastAPI(
    title="Simple Orchestration API",
    description="Three simple endpoints: ask-vedas, edumentor, wellness with GET and POST methods",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check data
health_data = {
    "startup_time": datetime.now(),
    "total_requests": 0,
    "successful_requests": 0
}

@app.get("/ask-vedas")
async def ask_vedas_get(
    query: str = Query(..., description="Your spiritual question"),
    user_id: str = Query("anonymous", description="User ID")
):
    return await process_vedas_query(query, user_id)

@app.post("/ask-vedas")
async def ask_vedas_post(request: QueryRequest):
    return await process_vedas_query(request.query, request.user_id)

async def process_vedas_query(query: str, user_id: str):
    try:
        sources = engine.search_documents(query, "vedas")
        context = "\n".join([doc["text"] for doc in sources[:2]])
        prompt = f"""You are a wise spiritual teacher. Based on ancient Vedic wisdom, provide profound guidance for this question: "{query}"

Context from sacred texts:
{context}

Provide spiritual wisdom that is authentic, practical, and inspiring. Keep it concise but meaningful."""
        fallback = f"The ancient Vedic texts teach us to seek truth through self-reflection and righteous action. Regarding '{query}', remember that true wisdom comes from understanding the interconnectedness of all existence. Practice mindfulness, act with compassion, and seek the divine within yourself."
        response_text, status = engine.generate_response(prompt, fallback, "vedas")
        return SimpleResponse(
            query_id=str(uuid.uuid4()),
            query=query,
            response=response_text,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            endpoint="ask-vedas",
            status=status
        )
    except Exception as e:
        logger.error(f"Error in ask-vedas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/edumentor")
async def edumentor_get(
    query: str = Query(..., description="Your learning question"),
    user_id: str = Query("anonymous", description="User ID")
):
    return await process_edumentor_query(query, user_id)

@app.post("/edumentor")
async def edumentor_post(request: QueryRequest):
    return await process_edumentor_query(request.query, request.user_id)

async def process_edumentor_query(query: str, user_id: str):
    try:
        sources = engine.search_documents(query, "educational")
        context = "\n".join([doc["text"] for doc in sources[:2]])
        prompt = f"""You are an expert educator. Explain this topic clearly and engagingly: "{query}"

Educational context:
{context}

Provide a clear, comprehensive explanation that:
- Uses simple, understandable language
- Includes practical examples
- Makes the topic interesting and memorable
- Is suitable for students"""
        fallback = f"Great question about '{query}'! This is an important topic to understand. Let me break it down for you in simple terms with practical examples that will help you learn and remember the key concepts. The main idea is to understand the fundamental principles and how they apply in real-world situations."
        response_text, status = engine.generate_response(prompt, fallback, "edumentor")
        return SimpleResponse(
            query_id=str(uuid.uuid4()),
            query=query,
            response=response_text,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            endpoint="edumentor",
            status=status
        )
    except Exception as e:
        logger.error(f"Error in edumentor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wellness")
async def wellness_get(
    query: str = Query(..., description="Your wellness concern"),
    user_id: str = Query("anonymous", description="User ID")
):
    return await process_wellness_query(query, user_id)

@app.post("/wellness")
async def wellness_post(request: QueryRequest):
    return await process_wellness_query(request.query, request.user_id)

async def process_wellness_query(query: str, user_id: str):
    try:
        sources = engine.search_documents(query, "wellness")
        context = "\n".join([doc["text"] for doc in sources[:2]])
        prompt = f"""You are a compassionate wellness counselor. Provide caring, helpful advice for: "{query}"

Wellness context:
{context}

Provide supportive guidance that:
- Shows empathy and understanding
- Offers practical, actionable advice
- Promotes overall wellbeing
- Is encouraging and positive"""
        fallback = f"Thank you for reaching out about '{query}'. It's important to take care of your wellbeing. Here are some gentle suggestions: Take time for self-care, practice deep breathing, stay connected with supportive people, and remember that small steps can lead to big improvements. If you're experiencing serious concerns, please consider speaking with a healthcare professional."
        response_text, status = engine.generate_response(prompt, fallback, "wellness")
        return SimpleResponse(
            query_id=str(uuid.uuid4()),
            query=query,
            response=response_text,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            endpoint="wellness",
            status=status
        )
    except Exception as e:
        logger.error(f"Error in wellness: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Simple Orchestration API",
        "version": "1.0.0",
        "endpoints": {
            "ask-vedas": {
                "GET": "/ask-vedas?query=your_question&user_id=optional",
                "POST": "/ask-vedas with JSON body"
            },
            "edumentor": {
                "GET": "/edumentor?query=your_question&user_id=optional", 
                "POST": "/edumentor with JSON body"
            },
            "wellness": {
                "GET": "/wellness?query=your_question&user_id=optional",
                "POST": "/wellness with JSON body"
            }
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Simple API."""
    try:
        # Update request count
        health_data["total_requests"] += 1

        # Check if engine is working
        engine_status = "healthy"
        try:
            # Test a simple search
            test_results = engine.search_documents("test", "educational")
            if not isinstance(test_results, list):
                engine_status = "degraded"
        except Exception as e:
            engine_status = f"unhealthy: {str(e)}"

        # Calculate uptime
        uptime_seconds = (datetime.now() - health_data["startup_time"]).total_seconds()

        return {
            "status": "healthy" if engine_status == "healthy" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime_seconds,
            "services": {
                "search_engine": engine_status,
                "llm_models": "healthy"  # Assuming models are working if engine works
            },
            "metrics": {
                "total_requests": health_data["total_requests"],
                "successful_requests": health_data["successful_requests"]
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="Simple Orchestration API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
    args = parser.parse_args()
    print("\n" + "="*60)
    print("  SIMPLE ORCHESTRATION API")
    print("="*60)
    print(f" Server URL: http://{args.host}:{args.port}")
    print(f" API Documentation: http://{args.host}:{args.port}/docs")
    print("\n Endpoints:")
    print("   GET/POST /ask-vedas - Spiritual wisdom")
    print("   GET/POST /edumentor - Educational content")
    print("   GET/POST /wellness - Health advice")
    print("="*60)
    uvicorn.run(app, host=args.host, port=args.port)



# import os
# import uuid
# import logging
# from datetime import datetime
# from typing import Optional, Dict, Any
# from pathlib import Path

# # FastAPI imports
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from contextlib import asynccontextmanager

# # LangChain imports
# from langchain_core.messages import HumanMessage
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # Environment and settings
# from dotenv import load_dotenv
# from config.settings import MODEL_CONFIG

# # Load environment variables
# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class ModelProvider:
#     """Model provider for Groq API using LangChain"""
#     def __init__(self, model_config: Dict[str, Any], endpoint: str):
#         self.model = None
#         self.model_name = model_config.get("model_name", "llama-3.1-8b-instruct")
#         self.api_key = os.getenv("GROQ_API_KEY")
#         self.backup_api_key = os.getenv("GROQ_API_KEY_BACKUP")
#         self.endpoint = endpoint
#         self.initialize_model()

#     def initialize_model(self):
#         """Initialize the Groq model"""
#         try:
#             if not self.api_key:
#                 raise ValueError("GROQ_API_KEY not found")
#             self.model = ChatGroq(
#                 model=self.model_name,
#                 api_key=self.api_key
#             )
#             logger.info(f"Initialized Groq model: {self.model_name} for {self.endpoint}")
#         except Exception as e:
#             logger.error(f"Failed to initialize model for {self.endpoint} with primary key: {e}")
#             if self.backup_api_key:
#                 try:
#                     logger.info(f"Trying backup API key for {self.endpoint}")
#                     self.model = ChatGroq(
#                         model=self.model_name,
#                         api_key=self.backup_api_key
#                     )
#                     logger.info(f"Initialized Groq model with backup key: {self.model_name}")
#                 except Exception as e:
#                     logger.error(f"Backup key failed for {self.endpoint}: {e}")
#                     self.model = None
#             else:
#                 self.model = None

#     def generate_response(self, prompt: str, fallback: str) -> str:
#         """Generate response using the Groq model"""
#         if not self.model:
#             logger.warning(f"No model initialized for {self.endpoint}. Using fallback response.")
#             return fallback
        
#         try:
#             response = self.model.invoke([HumanMessage(content=prompt)])
#             return response.content.strip()
#         except Exception as e:
#             logger.error(f"Model generation error for {self.endpoint}: {e}")
#             return fallback

# class SimpleOrchestrationEngine:
#     """Simple orchestration engine for the three main endpoints"""
    
#     def __init__(self):
#         self.vector_stores = {}
#         self.embedding_model = None
#         self.model_providers = {
#             "vedas": ModelProvider(MODEL_CONFIG["vedas_agent"], "ask-vedas"),
#             "edumentor": ModelProvider(MODEL_CONFIG["edumentor_agent"], "edumentor"),
#             "wellness": ModelProvider(MODEL_CONFIG["wellness_agent"], "wellness")
#         }
#         self.initialize_vector_stores()
        
#     def initialize_vector_stores(self):
#         """Initialize vector stores and embedding model"""
#         logger.info("Initializing embedding model...")
#         self.embedding_model = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )
        
#         # Load existing vector stores
#         vector_store_dir = Path("vector_stores")
#         store_names = ['vedas_index', 'wellness_index', 'educational_index', 'unified_index']
        
#         for store_name in store_names:
#             store_path = vector_store_dir / store_name
#             if store_path.exists():
#                 try:
#                     store = FAISS.load_local(
#                         str(store_path), 
#                         self.embedding_model, 
#                         allow_dangerous_deserialization=True
#                     )
#                     self.vector_stores[store_name.replace('_index', '')] = store
#                     logger.info(f"Loaded vector store: {store_name}")
#                 except Exception as e:
#                     logger.error(f"Failed to load vector store {store_name}: {e}")
        
#         logger.info(f"Initialized with {len(self.vector_stores)} vector stores")
    
#     def generate_response(self, prompt: str, fallback: str, endpoint: str) -> str:
#         """Generate response using the appropriate model provider"""
#         return self.model_providers[endpoint].generate_response(prompt, fallback)
    
#     def search_documents(self, query: str, store_type: str = "unified") -> list:
#         """Search relevant documents from vector store"""
#         if store_type in self.vector_stores:
#             try:
#                 retriever = self.vector_stores[store_type].as_retriever(search_kwargs={"k": 3})
#                 docs = retriever.invoke(query)
#                 return [{"text": doc.page_content[:500], "source": doc.metadata.get("source", "unknown")} for doc in docs]
#             except Exception as e:
#                 logger.error(f"Vector search error: {e}")
#         return []

# # Global engine instance
# engine = SimpleOrchestrationEngine()

# # Pydantic models
# class QueryRequest(BaseModel):
#     query: str
#     user_id: Optional[str] = "anonymous"

# class SimpleResponse(BaseModel):
#     query_id: str
#     query: str
#     response: str
#     sources: list
#     timestamp: str
#     endpoint: str

# # Lifespan handler
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     logger.info("Starting Simple Orchestration API...")
#     engine.initialize_vector_stores()
#     logger.info("Simple Orchestration API ready!")
    
#     yield
    
#     # Shutdown
#     logger.info("Shutting down Simple Orchestration API...")

# # Initialize FastAPI app
# app = FastAPI(
#     title="Simple Orchestration API",
#     description="Three simple endpoints: ask-vedas, edumentor, wellness with GET and POST methods",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==================== ASK-VEDAS ENDPOINTS ====================

# @app.get("/ask-vedas")
# async def ask_vedas_get(
#     query: str = Query(..., description="Your spiritual question"),
#     user_id: str = Query("anonymous", description="User ID")
# ):
#     """GET method for Vedas spiritual wisdom"""
#     return await process_vedas_query(query, user_id)

# @app.post("/ask-vedas")
# async def ask_vedas_post(request: QueryRequest):
#     """POST method for Vedas spiritual wisdom"""
#     return await process_vedas_query(request.query, request.user_id)

# async def process_vedas_query(query: str, user_id: str):
#     """Process Vedas query and return spiritual wisdom"""
#     try:
#         # Search relevant documents
#         sources = engine.search_documents(query, "vedas")
#         context = "\n".join([doc["text"] for doc in sources[:2]])
        
#         # Generate response
#         prompt = f"""You are a wise spiritual teacher. Based on ancient Vedic wisdom, provide profound guidance for this question: "{query}"

# Context from sacred texts:
# {context}

# Provide spiritual wisdom that is authentic, practical, and inspiring. Keep it concise but meaningful."""

#         fallback = f"The ancient Vedic texts teach us to seek truth through self-reflection and righteous action. Regarding '{query}', remember that true wisdom comes from understanding the interconnectedness of all existence. Practice mindfulness, act with compassion, and seek the divine within yourself."
        
#         response_text = engine.generate_response(prompt, fallback, "vedas")
        
#         return SimpleResponse(
#             query_id=str(uuid.uuid4()),
#             query=query,
#             response=response_text,
#             sources=sources,
#             timestamp=datetime.now().isoformat(),
#             endpoint="ask-vedas"
#         )
        
#     except Exception as e:
#         logger.error(f"Error in ask-vedas: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ==================== EDUMENTOR ENDPOINTS ====================

# @app.get("/edumentor")
# async def edumentor_get(
#     query: str = Query(..., description="Your learning question"),
#     user_id: str = Query("anonymous", description="User ID")
# ):
#     """GET method for educational content"""
#     return await process_edumentor_query(query, user_id)

# @app.post("/edumentor")
# async def edumentor_post(request: QueryRequest):
#     """POST method for educational content"""
#     return await process_edumentor_query(request.query, request.user_id)

# async def process_edumentor_query(query: str, user_id: str):
#     """Process educational query and return learning content"""
#     try:
#         # Search relevant documents
#         sources = engine.search_documents(query, "educational")
#         context = "\n".join([doc["text"] for doc in sources[:2]])
        
#         # Generate response
#         prompt = f"""You are an expert educator. Explain this topic clearly and engagingly: "{query}"

# Educational context:
# {context}

# Provide a clear, comprehensive explanation that:
# - Uses simple, understandable language
# - Includes practical examples
# - Makes the topic interesting and memorable
# - Is suitable for students"""

#         fallback = f"Great question about '{query}'! This is an important topic to understand. Let me break it down for you in simple terms with practical examples that will help you learn and remember the key concepts. The main idea is to understand the fundamental principles and how they apply in real-world situations."
        
#         response_text = engine.generate_response(prompt, fallback, "edumentor")
        
#         return SimpleResponse(
#             query_id=str(uuid.uuid4()),
#             query=query,
#             response=response_text,
#             sources=sources,
#             timestamp=datetime.now().isoformat(),
#             endpoint="edumentor"
#         )
        
#     except Exception as e:
#         logger.error(f"Error in edumentor: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ==================== WELLNESS ENDPOINTS ====================

# @app.get("/wellness")
# async def wellness_get(
#     query: str = Query(..., description="Your wellness concern"),
#     user_id: str = Query("anonymous", description="User ID")
# ):
#     """GET method for wellness advice"""
#     return await process_wellness_query(query, user_id)

# @app.post("/wellness")
# async def wellness_post(request: QueryRequest):
#     """POST method for wellness advice"""
#     return await process_wellness_query(request.query, request.user_id)

# async def process_wellness_query(query: str, user_id: str):
#     """Process wellness query and return health advice"""
#     try:
#         # Search relevant documents
#         sources = engine.search_documents(query, "wellness")
#         context = "\n".join([doc["text"] for doc in sources[:2]])
        
#         # Generate response
#         prompt = f"""You are a compassionate wellness counselor. Provide caring, helpful advice for: "{query}"

# Wellness context:
# {context}

# Provide supportive guidance that:
# - Shows empathy and understanding
# - Offers practical, actionable advice
# - Promotes overall wellbeing
# - Is encouraging and positive"""

#         fallback = f"Thank you for reaching out about '{query}'. It's important to take care of your wellbeing. Here are some gentle suggestions: Take time for self-care, practice deep breathing, stay connected with supportive people, and remember that small steps can lead to big improvements. If you're experiencing serious concerns, please consider speaking with a healthcare professional."
        
#         response_text = engine.generate_response(prompt, fallback, "wellness")
        
#         return SimpleResponse(
#             query_id=str(uuid.uuid4()),
#             query=query,
#             response=response_text,
#             sources=sources,
#             timestamp=datetime.now().isoformat(),
#             endpoint="wellness"
#         )
        
#     except Exception as e:
#         logger.error(f"Error in wellness: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ==================== ROOT ENDPOINT ====================

# @app.get("/")
# async def root():
#     """Root endpoint with API information"""
#     return {
#         "message": "Simple Orchestration API",
#         "version": "1.0.0",
#         "endpoints": {
#             "ask-vedas": {
#                 "GET": "/ask-vedas?query=your_question&user_id=optional",
#                 "POST": "/ask-vedas with JSON body"
#             },
#             "edumentor": {
#                 "GET": "/edumentor?query=your_question&user_id=optional", 
#                 "POST": "/edumentor with JSON body"
#             },
#             "wellness": {
#                 "GET": "/wellness?query=your_question&user_id=optional",
#                 "POST": "/wellness with JSON body"
#             }
#         },
#         "documentation": "/docs"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     import argparse

#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description="Simple Orchestration API")
#     parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
#     parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
#     args = parser.parse_args()

#     print("\n" + "="*60)
#     print("  SIMPLE ORCHESTRATION API")
#     print("="*60)
#     print(f" Server URL: http://{args.host}:{args.port}")
#     print(f" API Documentation: http://{args.host}:{args.port}/docs")
#     print("\n Endpoints:")
#     print("   GET/POST /ask-vedas - Spiritual wisdom")
#     print("   GET/POST /edumentor - Educational content")
#     print("   GET/POST /wellness - Health advice")
#     print("="*60)

#     uvicorn.run(app, host=args.host, port=args.port)




# """
# Simple FastAPI with 3 endpoints: ask-vedas, edumentor, wellness
# Each endpoint has both GET and POST methods for frontend integration
# """

# import os
# import uuid
# import logging
# from datetime import datetime
# from typing import Optional, Dict, Any
# from pathlib import Path

# # FastAPI imports
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from contextlib import asynccontextmanager

# # Environment and AI imports
# from dotenv import load_dotenv
# import google.generativeai as genai

# # LangChain imports
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # Load environment variables
# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class SimpleOrchestrationEngine:
#     """Simple orchestration engine for the three main endpoints"""
    
#     def __init__(self):
#         self.vector_stores = {}
#         self.embedding_model = None
#         self.gemini_model = None
#         self.initialize_gemini()
        
#     def initialize_gemini(self):
#         """Initialize Gemini API with failover"""
#         primary_key = os.getenv("GEMINI_API_KEY")
#         backup_key = os.getenv("GEMINI_API_KEY_BACKUP")
        
#         # Try primary key
#         if primary_key:
#             try:
#                 genai.configure(api_key=primary_key)
#                 self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
#                 test_response = self.gemini_model.generate_content("Hello")
#                 if test_response and test_response.text:
#                     logger.info("Gemini API initialized with primary key")
#                     return
#             except Exception as e:
#                 logger.warning(f"Primary Gemini API key failed: {e}")
        
#         # Try backup key
#         if backup_key:
#             try:
#                 genai.configure(api_key=backup_key)
#                 self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
#                 test_response = self.gemini_model.generate_content("Hello")
#                 if test_response and test_response.text:
#                     logger.info("Gemini API initialized with backup key")
#                     return
#             except Exception as e:
#                 logger.warning(f"Backup Gemini API key failed: {e}")
        
#         logger.error("Both Gemini API keys failed. Using fallback responses.")
#         self.gemini_model = None
    
#     def initialize_vector_stores(self):
#         """Initialize vector stores and embedding model"""
#         logger.info("Initializing embedding model...")
#         self.embedding_model = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )
        
#         # Load existing vector stores
#         vector_store_dir = Path("vector_stores")
#         store_names = ['vedas_index', 'wellness_index', 'educational_index', 'unified_index']
        
#         for store_name in store_names:
#             store_path = vector_store_dir / store_name
#             if store_path.exists():
#                 try:
#                     store = FAISS.load_local(
#                         str(store_path), 
#                         self.embedding_model, 
#                         allow_dangerous_deserialization=True
#                     )
#                     self.vector_stores[store_name.replace('_index', '')] = store
#                     logger.info(f"Loaded vector store: {store_name}")
#                 except Exception as e:
#                     logger.error(f"Failed to load vector store {store_name}: {e}")
        
#         logger.info(f"Initialized with {len(self.vector_stores)} vector stores")
    
#     def generate_response(self, prompt: str, fallback: str) -> str:
#         """Generate response using Gemini API with fallback"""
#         if self.gemini_model:
#             try:
#                 response = self.gemini_model.generate_content(prompt)
#                 if response and response.text:
#                     return response.text.strip()
#             except Exception as e:
#                 logger.warning(f"Gemini API error: {e}")
        
#         return fallback
    
#     def search_documents(self, query: str, store_type: str = "unified") -> list:
#         """Search relevant documents from vector store"""
#         if store_type in self.vector_stores:
#             try:
#                 retriever = self.vector_stores[store_type].as_retriever(search_kwargs={"k": 3})
#                 docs = retriever.invoke(query)
#                 return [{"text": doc.page_content[:500], "source": doc.metadata.get("source", "unknown")} for doc in docs]
#             except Exception as e:
#                 logger.error(f"Vector search error: {e}")
#         return []

# # Global engine instance
# engine = SimpleOrchestrationEngine()

# # Pydantic models
# class QueryRequest(BaseModel):
#     query: str
#     user_id: Optional[str] = "anonymous"

# class SimpleResponse(BaseModel):
#     query_id: str
#     query: str
#     response: str
#     sources: list
#     timestamp: str
#     endpoint: str

# # Lifespan handler
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     logger.info("Starting Simple Orchestration API...")
#     engine.initialize_vector_stores()
#     logger.info("Simple Orchestration API ready!")
    
#     yield
    
#     # Shutdown
#     logger.info("Shutting down Simple Orchestration API...")

# # Initialize FastAPI app
# app = FastAPI(
#     title="Simple Orchestration API",
#     description="Three simple endpoints: ask-vedas, edumentor, wellness with GET and POST methods",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==================== ASK-VEDAS ENDPOINTS ====================

# @app.get("/ask-vedas")
# async def ask_vedas_get(
#     query: str = Query(..., description="Your spiritual question"),
#     user_id: str = Query("anonymous", description="User ID")
# ):
#     """GET method for Vedas spiritual wisdom"""
#     return await process_vedas_query(query, user_id)

# @app.post("/ask-vedas")
# async def ask_vedas_post(request: QueryRequest):
#     """POST method for Vedas spiritual wisdom"""
#     return await process_vedas_query(request.query, request.user_id)

# async def process_vedas_query(query: str, user_id: str):
#     """Process Vedas query and return spiritual wisdom"""
#     try:
#         # Search relevant documents
#         sources = engine.search_documents(query, "vedas")
#         context = "\n".join([doc["text"] for doc in sources[:2]])
        
#         # Generate response
#         prompt = f"""You are a wise spiritual teacher. Based on ancient Vedic wisdom, provide profound guidance for this question: "{query}"

# Context from sacred texts:
# {context}

# Provide spiritual wisdom that is authentic, practical, and inspiring. Keep it concise but meaningful."""

#         fallback = f"The ancient Vedic texts teach us to seek truth through self-reflection and righteous action. Regarding '{query}', remember that true wisdom comes from understanding the interconnectedness of all existence. Practice mindfulness, act with compassion, and seek the divine within yourself."
        
#         response_text = engine.generate_response(prompt, fallback)
        
#         return SimpleResponse(
#             query_id=str(uuid.uuid4()),
#             query=query,
#             response=response_text,
#             sources=sources,
#             timestamp=datetime.now().isoformat(),
#             endpoint="ask-vedas"
#         )
        
#     except Exception as e:
#         logger.error(f"Error in ask-vedas: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ==================== EDUMENTOR ENDPOINTS ====================

# @app.get("/edumentor")
# async def edumentor_get(
#     query: str = Query(..., description="Your learning question"),
#     user_id: str = Query("anonymous", description="User ID")
# ):
#     """GET method for educational content"""
#     return await process_edumentor_query(query, user_id)

# @app.post("/edumentor")
# async def edumentor_post(request: QueryRequest):
#     """POST method for educational content"""
#     return await process_edumentor_query(request.query, request.user_id)

# async def process_edumentor_query(query: str, user_id: str):
#     """Process educational query and return learning content"""
#     try:
#         # Search relevant documents
#         sources = engine.search_documents(query, "educational")
#         context = "\n".join([doc["text"] for doc in sources[:2]])
        
#         # Generate response
#         prompt = f"""You are an expert educator. Explain this topic clearly and engagingly: "{query}"

# Educational context:
# {context}

# Provide a clear, comprehensive explanation that:
# - Uses simple, understandable language
# - Includes practical examples
# - Makes the topic interesting and memorable
# - Is suitable for students"""

#         fallback = f"Great question about '{query}'! This is an important topic to understand. Let me break it down for you in simple terms with practical examples that will help you learn and remember the key concepts. The main idea is to understand the fundamental principles and how they apply in real-world situations."
        
#         response_text = engine.generate_response(prompt, fallback)
        
#         return SimpleResponse(
#             query_id=str(uuid.uuid4()),
#             query=query,
#             response=response_text,
#             sources=sources,
#             timestamp=datetime.now().isoformat(),
#             endpoint="edumentor"
#         )
        
#     except Exception as e:
#         logger.error(f"Error in edumentor: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ==================== WELLNESS ENDPOINTS ====================

# @app.get("/wellness")
# async def wellness_get(
#     query: str = Query(..., description="Your wellness concern"),
#     user_id: str = Query("anonymous", description="User ID")
# ):
#     """GET method for wellness advice"""
#     return await process_wellness_query(query, user_id)

# @app.post("/wellness")
# async def wellness_post(request: QueryRequest):
#     """POST method for wellness advice"""
#     return await process_wellness_query(request.query, request.user_id)

# async def process_wellness_query(query: str, user_id: str):
#     """Process wellness query and return health advice"""
#     try:
#         # Search relevant documents
#         sources = engine.search_documents(query, "wellness")
#         context = "\n".join([doc["text"] for doc in sources[:2]])
        
#         # Generate response
#         prompt = f"""You are a compassionate wellness counselor. Provide caring, helpful advice for: "{query}"

# Wellness context:
# {context}

# Provide supportive guidance that:
# - Shows empathy and understanding
# - Offers practical, actionable advice
# - Promotes overall wellbeing
# - Is encouraging and positive"""

#         fallback = f"Thank you for reaching out about '{query}'. It's important to take care of your wellbeing. Here are some gentle suggestions: Take time for self-care, practice deep breathing, stay connected with supportive people, and remember that small steps can lead to big improvements. If you're experiencing serious concerns, please consider speaking with a healthcare professional."
        
#         response_text = engine.generate_response(prompt, fallback)
        
#         return SimpleResponse(
#             query_id=str(uuid.uuid4()),
#             query=query,
#             response=response_text,
#             sources=sources,
#             timestamp=datetime.now().isoformat(),
#             endpoint="wellness"
#         )
        
#     except Exception as e:
#         logger.error(f"Error in wellness: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # ==================== ROOT ENDPOINT ====================

# @app.get("/")
# async def root():
#     """Root endpoint with API information"""
#     return {
#         "message": "Simple Orchestration API",
#         "version": "1.0.0",
#         "endpoints": {
#             "ask-vedas": {
#                 "GET": "/ask-vedas?query=your_question&user_id=optional",
#                 "POST": "/ask-vedas with JSON body"
#             },
#             "edumentor": {
#                 "GET": "/edumentor?query=your_question&user_id=optional", 
#                 "POST": "/edumentor with JSON body"
#             },
#             "wellness": {
#                 "GET": "/wellness?query=your_question&user_id=optional",
#                 "POST": "/wellness with JSON body"
#             }
#         },
#         "documentation": "/docs"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     import argparse

#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description="Simple Orchestration API")
#     parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
#     parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
#     args = parser.parse_args()

#     print("\n" + "="*60)
#     print("  SIMPLE ORCHESTRATION API")
#     print("="*60)
#     print(f" Server URL: http://{args.host}:{args.port}")
#     print(f" API Documentation: http://{args.host}:{args.port}/docs")
#     print("\n Endpoints:")
#     print("   GET/POST /ask-vedas - Spiritual wisdom")
#     print("   GET/POST /edumentor - Educational content")
#     print("   GET/POST /wellness - Health advice")
#     print("="*60)

#     uvicorn.run(app, host=args.host, port=args.port)
