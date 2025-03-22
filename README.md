# RAG App Demo

This repository demonstrates a Retrieval-Augmented Generation (RAG) pipeline that integrates document retrieval with text generation to answer queries. It is designed as a demo application that showcases how to combine a collection of documents with a text generation model to produce context-aware answers.

## Overview

The RAG App works by:

- **Retrieving** the most relevant document from a collection of text files using embeddings computed by a Sentence Transformer.
- **Generating** an answer by feeding the retrieved document's content along with the user query into a Huggingface text generation model.
- Being **configurable** via a YAML configuration file, making it easy to update key parameters like model names, data paths, and generation settings.

Supported features include:

- An **interactive command-line interface** (`main.py`)
- A **FastAPI-based web API** (`api.py`)
- An optional **web scraper** (`scraper.py`)

## Getting Started

### Prerequisites

- Python 3.9 or later
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/julius-vr/RAG.git
   cd RAG
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

Edit the `config.yaml` file to set key parameters:

```yaml
data_path: "data"
cache_path: "cache/retriever_cache"
rebuild_index: false

embedding_model_name: "all-MiniLM-L12-v2"
generation_model_name: "google/flan-t5-large"
max_tokens: 200

logging:
  level: "INFO"
```

- `data_path`: Directory where your text documents are stored.
- `cache_path`: Location to store cached document embeddings.
- `rebuild_index`: Force re-computation of embeddings if `true`.
- `embedding_model_name`: Sentence Transformer for embeddings.
- `generation_model_name`: Huggingface model for response generation.
- `max_tokens`: Token limit for generated responses.
- `logging.level`: Controls logging verbosity.

## Populating the Data Store

Place plain text (`.txt`) documents into the `data/` folder. Each file should contain content used as context to answer queries.

You can also use the optional `scraper.py` script to fetch and save documents from predefined URLs.

## Usage

### CLI Demo

Run the interactive demo with:

```bash
python main.py
```

You'll see:

```
=== RAG App Demo ===
Type 'exit' to quit.

Enter your query:
```

Type a query, for example:

```
What are the main challenges in implementing quantum error correction for scalable quantum computing?
```

The system will:

1. Retrieve the most relevant document
2. Display its source and content
3. Construct a prompt with the query and the retrieved content
4. Generate and display the answer

### Web API

Start the API server:

```bash
uvicorn api:app --reload
```

Access it at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Send a POST request to `/query` with a JSON body:

```json
{
  "query": "What are the main challenges in implementing quantum error correction for scalable quantum computing?"
}
```

## Module Overview

- **`main.py`** – CLI interface that loads configuration, initializes the pipeline, and handles user queries interactively.
- **`api.py`** – FastAPI server exposing the pipeline via a `/query` endpoint.
- **`pipeline.py`** – Core logic that retrieves documents and generates answers.
- **`retriever.py`** – Loads documents, computes and caches embeddings, retrieves the most relevant one.
- **`generator.py`** – Uses Huggingface models to generate answers from prompts.
- **`scraper.py`** – (Optional) Fetches and saves documents from online sources.

## Troubleshooting

- **No documents retrieved?** Ensure the `data/` folder contains valid, non-empty `.txt` files.
- **Model loading issues?** Ensure internet access on first run to download models.
- **Configuration errors?** Double-check `config.yaml` for correct paths and model names.

## License

This project is licensed under the MIT License.
