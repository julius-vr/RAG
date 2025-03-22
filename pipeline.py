import logging
from retriever import DocumentRetriever
from generator import TextGenerator


class RAGPipeline:
    def __init__(self, config):
        logging.info("Initializing RAGPipeline with config: %s", config)
        # Initialize the retriever using configuration values
        self.retriever = DocumentRetriever(
            data_path=config["data_path"],
            cache_path=config["cache_path"],
            rebuild_index=config.get("rebuild_index", False),
            embedding_model_name=config["embedding_model_name"]
        )
        # Initialize the text generator using configuration values
        self.generator = TextGenerator(
            model_name=config["generation_model_name"],
            max_tokens=config["max_tokens"]
        )

    def run(self, query: str) -> str:
        logging.info("Running RAGPipeline for query: %s", query)
        # Retrieve the single most relevant document
        retrieved_doc = self.retriever.retrieve(query)
        if not retrieved_doc:
            logging.warning("No documents retrieved for query: %s", query)
            return "No relevant documents found."

        # Print which document was used for retrieval and its content
        print("----- Retrieved Document -----")
        print("Source:", retrieved_doc["source"])
        print("Content:")
        print(retrieved_doc["content"])
        print("------------------------------")

        # Build the prompt and generate the answer
        context = retrieved_doc["content"]
        prompt = self._build_prompt(query, context)
        answer = self.generator.generate(prompt)
        return answer

    def _build_prompt(self, query: str, context: str) -> str:
        prompt = (
            f"You are a knowledgeable assistant. Use the following context to answer the query.\n\n"
            f"Context:\n{context}\n\n"
            f"Query: {query}\n\n"
            f"Answer:"
        )
        logging.debug("Built prompt: %s", prompt)
        return prompt
