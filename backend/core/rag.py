import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGSystem, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found. RAG functionality will be limited.")
            return

        # Define paths
        project_root = settings.BASE_DIR.parent
        cache_dir = settings.BASE_DIR / "faiss_cache"
        cache_dir.mkdir(exist_ok=True)
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        self.retrievers = {}

        categories = {
            "thesis": project_root / "memoriatitulo.txt",
            "internship": project_root / "practicaprofesional.txt",
            "electives": project_root / "electivos.txt",
        }

        try:
            for cat, doc_path in categories.items():
                index_path = cache_dir / cat
                
                # Check if cached index exists
                if index_path.exists():
                    logger.info(f"Loading cached FAISS index for '{cat}'...")
                    vector_store = FAISS.load_local(
                        str(index_path), 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    # Create new index and cache it
                    logger.info(f"Creating FAISS index for '{cat}' (first time)...")
                    loader = TextLoader(str(doc_path), encoding="utf-8")
                    docs = loader.load()
                    vector_store = FAISS.from_documents(docs, self.embeddings)
                    vector_store.save_local(str(index_path))
                    logger.info(f"Cached FAISS index for '{cat}' at {index_path}")
                
                self.retrievers[cat] = vector_store.as_retriever(search_kwargs={"k": 3})
            
            self.initialized = True
            logger.info("RAG System initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG System: {e}")

    def query(self, category: str, query: str):
        if not self.initialized:
            self.initialize()
        
        if category not in self.retrievers:
            return "Category not found or RAG not initialized."
        
        docs = self.retrievers[category].invoke(query)
        return "\n\n".join([d.page_content for d in docs])

rag_system = RAGSystem()
