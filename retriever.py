import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import logging

class DocumentRetriever:
    def __init__(self, data_path, cache_path, rebuild_index, embedding_model_name):
        logging.info("Initializing DocumentRetriever")
        self.documents = []
        self.embeddings = None
        self.cache_path = cache_path

        # Load documents from the specified path
        self.load_documents(data_path)

        # Initialize the embedding model
        self.model = SentenceTransformer(embedding_model_name)
        logging.info("Using embedding model: %s", embedding_model_name)

        # Load cached embeddings if available; otherwise, compute them
        if self.documents:
            if os.path.exists(self.cache_path) and not rebuild_index:
                logging.info("Loading cached embeddings and document metadata")
                self.load_cache()
            else:
                logging.info("Computing embeddings for %d documents", len(self.documents))
                self.build_index()
                self.save_cache()
        else:
            logging.warning("No documents found in the specified data path.")

    def load_documents(self, data_path):
        txt_files = glob.glob(os.path.join(data_path, "*.txt"))
        for file in txt_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty documents
                        self.documents.append({
                            "content": content,
                            "source": file
                        })
                        logging.info("Loaded document: %s", file)
                    else:
                        logging.warning("File %s is empty. Skipping.", file)
            except Exception as e:
                logging.error("Error loading file %s: %s", file, str(e))

    def build_index(self):
        if not self.documents:
            logging.warning("No documents loaded. Skipping index build.")
            self.embeddings = None
            return
        # Compute embeddings for each document
        contents = [doc["content"] for doc in self.documents]
        self.embeddings = self.model.encode(contents, convert_to_numpy=True)
        # Normalize embeddings so that cosine similarity equals the dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        self.embeddings = self.embeddings / norms
        logging.info("Computed and normalized embeddings for %d documents", len(self.documents))

    def save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path + ".embeddings.pkl", "wb") as f:
                pickle.dump(self.embeddings, f)
            with open(self.cache_path + ".docs.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            logging.info("Saved embeddings and document metadata to cache")
        except Exception as e:
            logging.error("Error saving cache: %s", str(e))

    def load_cache(self):
        try:
            with open(self.cache_path + ".embeddings.pkl", "rb") as f:
                self.embeddings = pickle.load(f)
            with open(self.cache_path + ".docs.pkl", "rb") as f:
                self.documents = pickle.load(f)
            logging.info("Loaded cached embeddings and document metadata")
        except Exception as e:
            logging.error("Error loading cache: %s", str(e))
            self.embeddings = None

    def retrieve(self, query: str):
        logging.info("Retrieving the most similar document for query: %s", query)
        if self.embeddings is None or len(self.documents) == 0:
            logging.warning("No embeddings or documents available for retrieval.")
            return None
        # Compute query embedding and normalize it
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_norm = 1
        query_embedding = query_embedding / query_norm
        # Compute cosine similarity between query and all document embeddings
        cosine_similarities = np.dot(self.embeddings, query_embedding)
        # Retrieve the index of the most similar document
        best_idx = int(np.argmax(cosine_similarities))
        logging.info("Retrieved document index: %d", best_idx)
        return self.documents[best_idx]
