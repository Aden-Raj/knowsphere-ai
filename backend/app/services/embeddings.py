from functools import cached_property
from pathlib import Path
from threading import Lock


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32,
                 cache_folder: Path | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_folder = cache_folder
        self._inference_lock = Lock()

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=str(self.cache_folder) if self.cache_folder else None,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._inference_lock:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device,
            )
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors
