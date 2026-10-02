"""
Advanced example showing additional features like:
- Saving/loading indexes
- Batch processing queries
- Custom query settings
"""

import os
import pickle
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class DocumentSearchRAG:
    """Advanced RAG document search class."""
    
    def __init__(self, documents_folder="documents", index_file="index.pkl"):
        self.documents_folder = documents_folder
        self.index_file = index_file
        self.index = None
        self.query_engine = None
        self._initialize()
    
    def _initialize(self):
        """Initialize LLM and embeddings."""
        llm = Ollama(model="qwen3:4b", base_url="http://localhost:11434")
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
    
    def load_documents(self):
        """Load documents from folder."""
        documents_path = Path(self.documents_folder)
        documents_path.mkdir(parents=True, exist_ok=True)
        
        reader = SimpleDirectoryReader(
            input_dir=str(documents_path),
            required_exts=[".pdf", ".txt"],
            recursive=True,
        )
        documents = reader.load_data()
        return documents
    
    def create_index(self, force_rebuild=False):
        """Create index, either from saved file or from documents."""
        # Try to load saved index
        if not force_rebuild and Path(self.index_file).exists():
            print(f"Loading saved index from {self.index_file}")
            with open(self.index_file, "rb") as f:
                self.index = pickle.load(f)
            self.query_engine = self.index.as_query_engine()
            return
        
        # Create new index
        print("Creating new index...")
        documents = self.load_documents()
        if not documents:
            print("No documents found!")
            return
        
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_query_engine()
        
        # Save index
        self.save_index()
    
    def save_index(self):
        """Save index to disk."""
        print(f"Saving index to {self.index_file}")
        with open(self.index_file, "wb") as f:
            pickle.dump(self.index, f)
    
    def query(self, query_text):
        """Execute a query."""
        if self.query_engine is None:
            print("Index not created. Please create index first.")
            return None
        
        response = self.query_engine.query(query_text)
        return response
    
    def batch_queries(self, queries):
        """Execute multiple queries and return results."""
        results = {}
        for query in queries:
            print(f"\nQuery: {query}")
            response = self.query(query)
            results[query] = str(response)
            print(f"Answer: {response}\n")
        return results


def main_advanced():
    """Advanced example usage."""
    print("Advanced RAG Document Search Example")
    print("=" * 60)
    
    # Initialize RAG system
    rag = DocumentSearchRAG()
    
    # Create index (loads from disk if exists, otherwise builds from documents)
    rag.create_index(force_rebuild=False)
    
    if rag.query_engine is None:
        print("Could not create index. Exiting.")
        return
    
    # Example 1: Single query
    print("\nExample 1: Single Query")
    print("-" * 60)
    result = rag.query("What are the main topics in the documents?")
    print(f"Answer: {result}\n")
    
    # Example 2: Batch queries
    print("Example 2: Batch Queries")
    print("-" * 60)
    queries = [
        "Summarize the key points",
        "What data is available?",
        "What conclusions are drawn?",
    ]
    results = rag.batch_queries(queries)
    
    # Example 3: Interactive session
    print("\nExample 3: Interactive Session")
    print("-" * 60)
    print("Type 'exit' to quit interactive mode\n")
    
    while True:
        user_query = input("Ask a question: ").strip()
        if user_query.lower() in ["exit", "quit", "q"]:
            break
        if user_query:
            response = rag.query(user_query)
            print(f"Answer: {response}\n")


if __name__ == "__main__":
    main_advanced()
