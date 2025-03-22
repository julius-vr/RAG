from transformers import pipeline
import logging

class TextGenerator:
    def __init__(self, model_name="google/flan-t5-large", max_tokens=200):
        logging.info("Loading Huggingface text generation model: %s", model_name)
        self.generator = pipeline("text2text-generation", model=model_name)
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        logging.info("Generating response using model: %s", self.generator.model.name_or_path)
        outputs = self.generator(prompt, max_length=self.max_tokens, truncation=True)
        generated_text = outputs[0]['generated_text'].strip()
        logging.debug("Generated text: %s", generated_text)
        return generated_text
