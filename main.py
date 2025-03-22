import logging
import yaml
import os
from pipeline import RAGPipeline


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def configure_logging(level_str):
    numeric_level = getattr(logging, level_str.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level,
                        format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    config = load_config()
    configure_logging(config.get("logging", {}).get("level", "INFO"))

    # Ensure the data path exists
    data_path = config["data_path"]
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        logging.info("Created data folder at: %s", data_path)

    # Initialize the RAG pipeline with configuration
    pipeline = RAGPipeline(config=config)

    print("=== RAG App ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = pipeline.run(query)
        print("\nAnswer:\n", answer, "\n")


if __name__ == "__main__":
    main()

# query: "What are the main challenges in implementing quantum error correction for scalable quantum computing?"