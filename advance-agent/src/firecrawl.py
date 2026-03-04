import os
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()


class FirecrawlService:
    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("Missing FIRECRAWL_API_KEY environment variable")
        self.app = FirecrawlApp(api_key=api_key)

    def search_companies(self, query: str, num_results: int = 5):
        try:
            result = self.app.search(
                query=f"{query} alternatives comparison",
                limit=num_results
            )

            # Firecrawl returns an object → actual results are in result.data
            return result.data if hasattr(result, "data") else []

        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def scrape_company_pages(self, url: str):
        try:
            result = self.app.scrape(
                url=url,
                formats=["markdown"]
            )
            return result
        except Exception as e:
            print(f"Scrape failed: {e}")
            return None