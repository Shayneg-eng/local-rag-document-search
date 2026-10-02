import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


def initialize_llm():
    """Initialize the Ollama LLM with qwen3:4b model."""
    llm = Ollama(model="qwen3:4b", base_url="http://localhost:11434")
    return llm


def initialize_embeddings():
    """Initialize embeddings using Ollama."""
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    return embed_model


def load_documents(documents_folder="documents"):
    """Load documents from the specified folder."""
    documents_path = Path(documents_folder)
    
    if not documents_path.exists():
        print(f"Creating documents folder at {documents_path.absolute()}")
        documents_path.mkdir(parents=True, exist_ok=True)
        return []
    
    # Check if there are any documents
    pdf_files = list(documents_path.glob("*.pdf"))
    txt_files = list(documents_path.glob("*.txt"))
    
    if not pdf_files and not txt_files:
        print(f"No documents found in {documents_path.absolute()}")
        print("Please place PDF or TXT files in the documents folder.")
        return []
    
    print(f"Loading documents from {documents_path.absolute()}")
    reader = SimpleDirectoryReader(
        input_dir=str(documents_path),
        required_exts=[".pdf", ".txt"],
        recursive=True,
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")
    return documents


def create_index(documents):
    """Create a vector store index from documents."""
    if not documents:
        return None
    
    print("Creating index... This may take a moment.")
    index = VectorStoreIndex.from_documents(documents)
    print("Index created successfully")
    return index


def main():
    """Main function to run the RAG document search application."""
    print("=" * 60)
    print("RAG Document Search with LlamaIndex and Ollama")
    print("=" * 60)
    
    # Initialize LLM and embeddings
    print("\nInitializing Ollama LLM (qwen3:4b)...")
    llm = initialize_llm()
    embed_model = initialize_embeddings()
    
    # Configure settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    documents = load_documents()
    
    if not documents:
        print("\nExiting: No documents to search.")
        return
    
    # Create index
    index = create_index(documents)
    
    if index is None:
        print("\nExiting: Could not create index.")
        return
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    # Interactive query loop
    print("\n" + "=" * 60)
    print("Ready to answer questions about your documents!")
    print("Type 'exit' to quit")
    print("=" * 60 + "\n")
    
    while True:
        user_query = input("Ask a question: ").strip()
        
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        if not user_query:
            print("Please enter a question.\n")
            continue
        
        print("\nSearching documents...\n")
        response = query_engine.query(user_query)
        print(f"Answer: {response}\n")


if __name__ == "__main__":
    main()
