"""
Skill2Vec Embeddings

Learns dense vector representations of skills based on their
co-occurrence in job postings (analogous to Word2Vec over skill sequences).

Requires the ``gensim`` library (optional dependency).
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def train_skill_embeddings(
    skill_sequences: List[List[str]],
    vector_size: int = 64,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 30,
) -> Dict[str, List[float]]:
    """
    Train Skill2Vec embeddings.

    Args:
        skill_sequences: list of skill lists (one list per job posting).
        vector_size: embedding dimensionality.
        window: context window for Word2Vec.
        min_count: ignore skills appearing fewer times.
        epochs: training epochs.

    Returns:
        Mapping of skill name -> embedding vector.
    """

    try:
        from gensim.models import Word2Vec
    except ImportError:
        logger.warning("gensim not installed -- returning empty embeddings")
        return {}

    if not skill_sequences:
        return {}

    model = Word2Vec(
        sentences=skill_sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
    )

    embeddings = {
        word: model.wv[word].tolist() for word in model.wv.index_to_key
    }
    logger.info("Trained embeddings for %d skills (dim=%d)", len(embeddings), vector_size)
    return embeddings


def most_similar(
    embeddings: Dict[str, List[float]],
    skill: str,
    top_n: int = 10,
) -> List[str]:
    """Return the top-N most similar skills by cosine distance."""

    import numpy as np

    if skill not in embeddings:
        return []

    target = np.array(embeddings[skill])
    scored = []
    for other, vec in embeddings.items():
        if other == skill:
            continue
        v = np.array(vec)
        cos = np.dot(target, v) / (np.linalg.norm(target) * np.linalg.norm(v) + 1e-9)
        scored.append((other, float(cos)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n]]
