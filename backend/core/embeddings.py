from typing import List
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from core import get_logger
from config import get_settings

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    model_name: str
    device: str
    batch_size: int


class EmbeddingPipeline:
    """Batch embedding for TextChunks and queries."""

    def __init__(self):
        settings = get_settings()

        self.config = EmbeddingConfig(
            model_name=settings.embedding_model,
            device="cpu",  # or make this configurable later if you add ENV
            batch_size=settings.embedding_batch_size,
        )

        self.model = SentenceTransformer(self.config.model_name)
        self.model.to(self.config.device)
        logger.info(
            f"EmbeddingPipeline ready: {self.config.model_name} on {self.config.device}, "
            f"batch_size={self.config.batch_size}"
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return embeddings as numpy array [N, D]."""
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        emb = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.config.device,
        )
        return emb

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
