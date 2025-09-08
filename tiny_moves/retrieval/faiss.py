import json
import pickle
from pathlib import Path
from typing import Generic, TypeVar

import faiss
from faiss import IndexFlatIP
from sentence_transformers import SentenceTransformer

from tiny_moves.retrieval.metadata import Metadata

# Generic metadata type variable
M = TypeVar("M", bound=Metadata)


class Faiss(Generic[M]):
    """Abstract base class for constructing FAISS indices."""

    def __init__(self, input_dir: Path, metadata_cls: type[M]):
        """
        Initialize the FAISS constructor from a supplied directory.

        Note that different indices may require different MetaData classes. The MetaData
        class defines what sort of information may be returned to the queries.

        Args:
            input_dir: Directory containing the faiss index and metadata files
            metadata_cls: the class for the MetaData object

        """
        self.input_dir = input_dir
        self.embedding_model: SentenceTransformer | None = None
        self.index: IndexFlatIP | None = None
        self.metadata_cls = metadata_cls
        self.metadata: list[M] | None = None
        self.text_lookup: dict[str, str] | None = None

    def load_index(self) -> None:
        """Load FAISS index, metadata, optional text_lookup, and manifest."""
        index_path = self.input_dir / "faiss.index"
        metadata_path = self.input_dir / "metadata.pkl"
        lookup_path = self.input_dir / "text_lookup.pkl"
        manifest_path = self.input_dir / "manifest.json"

        # Load index
        self.index = faiss.read_index(index_path.as_posix())

        # Load metadata
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # Load optional text_lookup
        if lookup_path.exists():
            with open(lookup_path, "rb") as f:
                self.text_lookup = pickle.load(f)

        # Load manifest
        with open(manifest_path) as mf:
            manifest = json.load(mf)
        model_name = manifest["model_name"]

        self.embedding_model = SentenceTransformer(model_name)

    def search_index(self, query: str, top_k: int = 20) -> list[tuple[M, float]]:
        """
        Search the FAISS index for the given query string.

        Returns a list of top metadata results for a query.

        """
        if self.metadata is None or self.embedding_model is None or self.index is None:
            self.load_index()
            assert self.metadata is not None
            assert self.embedding_model is not None
            assert self.index is not None

        q_vec = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        scores, indices = self.index.search(q_vec, top_k)
        return [(self.metadata[i], score) for score, i in zip(scores[0], indices[0])]

    def get_chunk_text_context(self, key: str) -> str:
        """
        Get the chunk text context for a given key.

        Note that only FAISS indices where the MetaData has a text_lookup entry are supported.
        """
        if self.text_lookup is None:
            raise RuntimeError("text_lookup not loaded.")
        return self.text_lookup[key]


def get_faiss_index(cls: type[M], path: Path) -> Faiss[M]:
    """Retrieve the Faiss class with the correct metadata type."""
    return Faiss(path, cls)
