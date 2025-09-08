import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


def embed_text_with_sentence_transformer(
    text: str, model: SentenceTransformer, max_tokens: int = 500
) -> NDArray[np.float_]:
    """
    Embed a text string using a SentenceTransformer model, handling long texts by splitting them into chunks.

    Args:
        text: The input text to embed.
        model: A SentenceTransformer model used for embedding the text.
        max_tokens: The maximum number of tokens per chunk. Default is 500.

    Returns:
        np.ndarray: The averaged embedding of the text.

    """
    words = text.split()
    if len(words) <= max_tokens:
        emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb

    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk_words = words[i : i + max_tokens]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

    chunk_embs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    avg_emb = np.mean(chunk_embs, axis=0)
    avg_emb: NDArray[np.float_] = avg_emb / np.linalg.norm(avg_emb)
    return avg_emb
