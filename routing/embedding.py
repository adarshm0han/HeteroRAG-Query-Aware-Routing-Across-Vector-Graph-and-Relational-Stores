"""Lazy, cached local BGE query encoding; no hosted embedding calls or downloads."""

import os
from time import perf_counter

from baseline.common import MODEL, QUERY_PREFIX


class LocalEmbedding:
    def __init__(self, cache):
        self.cache = cache
        self.model = None

    def encode(self, question):
        start = perf_counter()
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(min(8, os.cpu_count() or 1))
            self.model = SentenceTransformer(
                MODEL, cache_folder=str(self.cache), device="cpu", local_files_only=True
            )
            self.model.max_seq_length = 512
        vector = self.model.encode(
            QUERY_PREFIX + question, normalize_embeddings=True, convert_to_numpy=True
        )
        return vector, perf_counter() - start
