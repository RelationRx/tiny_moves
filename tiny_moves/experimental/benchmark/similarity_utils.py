from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_embedding_similarity(
    reference_texts: list[str], component_texts: list[str], model: SentenceTransformer
) -> Any:
    """
    Compute cosine similarity matrix.

    Comparison: reference statements and hypotheses components using SentenceTransformer.
    """
    safe_refs = [t if t.strip() else "[EMPTY]" for t in reference_texts]
    safe_comps = [t if t.strip() else "[EMPTY]" for t in component_texts]
    ref_embeddings = model.encode(safe_refs, convert_to_tensor=True)
    comp_embeddings = model.encode(safe_comps, convert_to_tensor=True)
    sim_matrix = util.cos_sim(comp_embeddings, ref_embeddings).cpu().numpy()
    return np.nan_to_num(sim_matrix)


def compute_tfidf_similarity_matrix(refs: list[str], texts: list[str]) -> Any:
    """Compute TF-IDF-based semantic similarity per text."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(refs + texts)
    return cosine_similarity(tfidf_matrix[len(refs) :], tfidf_matrix[: len(refs)])


def extract_keywords(text: str) -> set[str]:
    """Extract keyword-based concepts, naive."""
    return set(text.lower().split())


def compute_keyword_similarity_matrix(refs: list[str], texts: list[str]) -> Any:
    """
    Compute keyword-based semantic similarity matrix.

    Comparison: reference statements and hypotheses components.
    """
    ref_keywords = [extract_keywords(ref) for ref in refs]
    text_keywords = [extract_keywords(txt) for txt in texts]
    matrix = []
    for txt_kw in text_keywords:
        row = []
        for ref_kw in ref_keywords:
            sim = len(txt_kw & ref_kw) / len(txt_kw | ref_kw) if ref_kw and txt_kw else 0
            row.append(sim)
        matrix.append(row)
    return np.array(matrix)
