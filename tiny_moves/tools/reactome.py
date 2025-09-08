from pathlib import Path

from tiny_moves.retrieval.faiss import get_faiss_index
from tiny_moves.retrieval.metadata import PlainTextMetadata

from ._registry import register_function_for_tool

REACTOME_INDEX_DIR = Path("/cache-juicefs/tiny_moves_data/faiss_indices/reactome_corpus/")
REACTOME_INDEX = get_faiss_index(PlainTextMetadata, REACTOME_INDEX_DIR)


@register_function_for_tool("reactome")
def search_for_evidence(query_string: str, threshold: float = 0.0) -> set[str]:
    """
    Search a database of scientific reports.

    Be specific as to the particular aspect of a hypothesis you would like evidence for.

    Be specific as to the particular type of evidence you would like.


    Args:
        query_string (str): The query string to search for related hypotheses.
        evidence_span (str): The span of evidence to return. Can be 'chunk' or 'section'.
            'chunk' returns the chunk text, while 'section' returns the section title.
        threshold (float): The minimum score threshold for a report to be considered relevant. Use default.

    Returns:
        list[dict[str,str]]: A list of dictionaries, with the key being the gene whose report the evidence is
        returned from, and the value is the snippet of text that matched the query

    """
    results = REACTOME_INDEX.search_index(query_string)

    evidence = [
        REACTOME_INDEX.get_chunk_text_context(result.chunk_id) for result, score in results if score > threshold
    ]

    # multiple chunks might come from the same paragraph/section
    unique_evidence = set(evidence)

    return unique_evidence
