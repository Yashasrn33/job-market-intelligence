from typing import Dict, Generator, Optional


class JSearchScraper:
    """
    Placeholder scraper for RapidAPI JSearch (or similar).
    Implement when credentials and API contract are defined.
    """

    def __init__(self, api_key: str, base_url: str = ""):
        self.api_key = api_key
        self.base_url = base_url

    def scrape(self) -> Generator[Dict, None, None]:
        if False:  # pragma: no cover
            yield {}
        return

