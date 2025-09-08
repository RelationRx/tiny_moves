from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from tiny_moves.experimental.benchmark.similarity_utils import (
    compute_embedding_similarity,
    compute_keyword_similarity_matrix,
    compute_tfidf_similarity_matrix,
)


def compute_hypothesis_scores(
    reference_statements: list[str],
    hypotheses_dict: dict[Any, Any],
    gene_name: str,
    model: SentenceTransformer,
    alpha: float = 0.33,
    beta: float = 0.33,
    gamma: float = 0.34,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce all scores for hypotheses."""
    detailed_rows = []
    summary_rows = []

    hypotheses = hypotheses_dict.get("hypotheses", [])
    if not hypotheses:
        raise ValueError(f"No hypotheses found for gene '{gene_name}'")

    for h in hypotheses:
        hyp_text = h["hypothesis"]
        comps = [c["biological_process"] for c in h.get("biological_processes", [])]
        if not comps:
            continue

        tfidf_sim = compute_tfidf_similarity_matrix(reference_statements, comps)
        emb_sim = compute_embedding_similarity(reference_statements, comps, model)
        kw_sim = compute_keyword_similarity_matrix(reference_statements, comps)

        for i, comp in enumerate(comps):
            for j, ref in enumerate(reference_statements):
                detailed_rows.append(
                    {
                        "gene": gene_name,
                        "hypothesis": hyp_text,
                        "biological_process": comp,
                        "reference_statement": ref,
                        "semantic similarity score (tfidf)": tfidf_sim[i, j],
                        "concept coverage score": kw_sim[i, j],
                        "embedding similarity score": emb_sim[i, j],
                    }
                )

            tfidf_best = np.max(tfidf_sim[i])
            emb_best = np.max(emb_sim[i])
            kw_best = np.max(kw_sim[i])

            final = alpha * tfidf_best + beta * kw_best + gamma * emb_best
            summary_rows.append(
                {
                    "gene": gene_name,
                    "hypothesis": hyp_text,
                    "biological_process": comp,
                    "semantic similarity score (tfidf)": tfidf_best,
                    "concept coverage score": kw_best,
                    "embedding similarity score": emb_best,
                    "process score": final,
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    df_detailed = pd.DataFrame(detailed_rows)

    if not df_summary.empty:
        a_df = df_summary.groupby(["gene", "hypothesis"], as_index=False)["process score"].mean()
        a_df.rename(columns={"process score": "final hypothesis score"}, inplace=True)
        df_summary = df_summary.merge(a_df, on=["gene", "hypothesis"])

    return df_summary, df_detailed
