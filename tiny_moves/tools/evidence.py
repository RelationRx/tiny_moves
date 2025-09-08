import os
from pathlib import Path
from typing import Any

from tiny_moves.retrieval.faiss import Faiss, get_faiss_index
from tiny_moves.retrieval.metadata import PlainTextMetadata


from ._registry import register_function_for_tool

REPORT_INDICES: dict[str, Faiss[Any]] = {}



def search_index(INDEX: Faiss[Any], query_string: str, threshold: float = 0.5) -> set[str]:
    """
    Search a database of gene-level reports for evidence relating to a specific aspect of a hypothesis.

    Be specific as to the particular aspect of a hypothesis you would like evidence for.

    Be specific as to the particular type of evidence you would like.


    Args:
        INDEX (Faiss): The Faiss index to search. This should be a pre-loaded index of reports.
        query_string (str): The query string to search for related hypotheses.
        threshold (float): The minimum score threshold for a report to be considered relevant. Use default.

    Returns:
        set[str]: A set of text snippets that matched the query.

    """
    results = INDEX.search_index(query_string)

    evidence = [INDEX.get_chunk_text_context(result.chunk_id) for result, score in results if score > threshold]

    # multiple chunks might come from the same paragraph/section
    unique_evidence = set(evidence)

    return unique_evidence



@register_function_for_tool("corpus")
def search_corpus(query_string: str, threshold: float = 0.5) -> set[str]:
    """
    Search the corpus database for evidence relating to a specific aspect of a hypothesis.

    Be specific as to the particular aspect of a hypothesis you would like evidence for.

    Formulate the query as a natural language question or statement rather than keywords.

    Args:
        query_string (str): The query string to search for related hypotheses.
        threshold (float): The minimum score threshold for a report to be considered relevant. Use default.

    Returns:
        set[str]: A set of text snippets that matched the query.

    """
    if "corpus" not in REPORT_INDICES:
        try:
            REPORT_INDEX_DIR = Path(os.environ["INDEX_DIR"])
        except KeyError:
            raise ValueError(
                "Environment variable INDEX_DIR is not set. Please set it to the directory containing the corpus index."
            )

        REPORT_INDEX = get_faiss_index(PlainTextMetadata, REPORT_INDEX_DIR)

        REPORT_INDICES["corpus"] = REPORT_INDEX

    return search_index(REPORT_INDICES["corpus"], query_string, threshold)
