import requests
from bs4 import BeautifulSoup
import logging

class WebScraper:
    def __init__(self):
        pass

    def scrape_page(self, url: str) -> str:
        logging.info("Scraping URL: %s", url)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Attempt to extract content from <article> tags first
                articles = soup.find_all('article')
                if articles:
                    text = "\n".join([article.get_text(separator="\n", strip=True) for article in articles])
                else:
                    # Fallback: extract text from all <p> tags
                    paragraphs = soup.find_all('p')
                    text = "\n".join([p.get_text(strip=True) for p in paragraphs])
                return text
            else:
                logging.error("Failed to retrieve URL: %s with status code: %d", url, response.status_code)
                return ""
        except Exception as e:
            logging.error("Exception occurred while scraping URL %s: %s", url, str(e))
            return ""

    def scrape_and_save(self, url: str, output_path: str):
        text = self.scrape_page(url)
        if text:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            logging.info("Saved scraped data to %s", output_path)
        else:
            logging.warning("No text scraped from URL: %s", url)
