# RAG Document Search with LlamaIndex and Ollama

A Python application that allows you to interact with PDF and TXT documents using a local LLM (Large Language Model) via Ollama. Uses LlamaIndex for document indexing and retrieval.

## Prerequisites

1. **Ollama** - Download and install from [ollama.ai](https://ollama.ai)
2. **Python 3.8+**
3. **qwen3:4b model** - Pull it with: `ollama pull qwen3:4b`

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```
   (Keep this running in a separate terminal)

3. **Place your documents:**
   - Put PDF and TXT files in the `documents/` folder
   - The application will automatically discover and load them

## Usage

Run the application:
```bash
python main.py
```

The application will:
1. Load all PDF and TXT files from the `documents/` folder
2. Create a vector index for fast retrieval
3. Start an interactive session where you can ask questions about your documents

Example:
```
Ask a question: What is the main topic discussed in the documents?
Searching documents...
Answer: The main topic is about...
```

## How It Works

1. **Document Loading**: The application reads all PDF and TXT files from the `documents/` folder
2. **Indexing**: LlamaIndex creates a vector store index using Ollama embeddings
3. **Retrieval**: When you ask a question, the system retrieves relevant document chunks
4. **Generation**: The Ollama qwen3:4b model generates an answer based on the retrieved context

## Features

- Supports PDF and TXT files
- Fast vector-based document retrieval
- Interactive command-line interface
- Uses local Ollama LLM (no external API calls)
- Easy to add more documents (just place them in the `documents/` folder)

## Troubleshooting

**Error: "Connection refused" when connecting to Ollama**
- Make sure Ollama server is running (`ollama serve`)
- Default URL is `http://localhost:11434`

**Model not found error**
- Pull the model: `ollama pull qwen3:4b`
- Ensure you're using the correct model name

**No documents found**
- Check that PDF and TXT files are in the `documents/` folder
- Ensure file extensions are `.pdf` or `.txt` (lowercase)

## Model Options

You can change the model used by modifying the model name in `main.py`:
- `qwen3:4b` - Default, lightweight model
- `qwen3:7b` - Larger, more capable
- `llama2` - Meta's Llama model
- Others available via `ollama pull <model>`

Just change the model name in the `initialize_llm()` and `initialize_embeddings()` functions.

## Advanced Usage

If you want to modify the application:

- **Change chunk size**: Modify `Settings` in `main.py`
- **Use different retriever**: Replace `QueryEngine` with custom retriever
- **Add similarity scores**: Modify the query engine to show relevance scores
- **Save index**: Persist the index to disk for faster loading

## License

Free to use and modify for your needs.
