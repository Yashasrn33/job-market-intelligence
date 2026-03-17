from typing import Dict, Generator


class KaggleLoader:
    """
    Placeholder loader for Kaggle datasets.
    Intended for offline ingestion into the same raw schema.
    """

    def __init__(self, dataset: str):
        self.dataset = dataset

    def load(self) -> Generator[Dict, None, None]:
        if False:  # pragma: no cover
            yield {}
        return

