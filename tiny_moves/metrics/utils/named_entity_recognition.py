from typing import Literal

import gilda
import nltk
from nltk.corpus import stopwords

BANNED_ENTITIES: set[str] = {
    "HGNC_7094_MICE",
    "MESH_D013534_Survival",
    "MESH_D006801_Humans",
    "MESH_D000339_Affect",
    "CHEBI_CHEBI:50906_role",
    "MESH_D016454_Review",
    "CHEBI_CHEBI:141213_isoxaflutole",
    "MESH_D005246_Feedback",
    "GO_GO:0008150_biological_process",
    "EFO_0001461_control",
    "GO_GO:0001897_cytolysis by symbiont of host cells",
    "MESH_D012106_Research",
    "MESH_D014024_Tissues",
    "MESH_D014157_Transcription Factors",
    "GO_GO:0007165_signal transduction",
    "GO_GO:0023052_signaling",
    "MESH_D020478_Form",
    "CHEBI_CHEBI:34922_picloram",
    "GO_GO:0005622_intracellular anatomical structure",
    "MESH_D001244_Association",
    "CHEBI_CHEBI:25212_metabolite",
    "MESH_D004798_Enzymes",
}


def extract_and_filter_biomedical_entities(
    text: str,
    replace: list[str] | None = ["/", "-", "–", "—"],  # noqa
    type_subsets: list[Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"]] | None = None,
    return_identifier: bool = False,
) -> set[str]:
    """
    Run Gilda NER with optional token replacement, namespace filtering, and ID extraction.

    Note: high confidence types are HGNC and GO.

    This function wraps a single-pass Gilda extraction (via `single_pass_extract_biomedical_entities`)
    and adds three optional steps:
      1. Token replacement: for each character in `replace` (e.g., "-" or "/"), replace it
         with a space in `text` and rerun the extraction. Useful when hyphens/slashes
         prevent correct recognition (e.g., "BRCA1/BRCA2").
      2. Namespace filtering: keep only identifiers whose string starts with one of
         the prefixes in `type_subsets` (e.g., ["HGNC", "CHEBI"]). If None, no filtering.
      3. Return ID-only: if `return_identifier` is True, strip off "<DB>_" and "_<entry_name>"
         and return only the middle "<ID>" segment (e.g., "11998" from "HGNC_11998_TP53").

    Args:
        text: Free-form text in which to locate and ground biomedical entities.
        replace: List of single-character strings to replace with a space before re-extraction.
                 If None, no second pass is done.
        type_subsets: List of ontology namespaces to keep (e.g., ["HGNC", "MESH"]).
                      Only identifiers starting with those prefixes are retained. If None,
                      all extracted identifiers are kept.
        return_identifier: If True, return only the accession portion ("<ID>") of each match.
                           If False, return full "<DB>_<ID>_<entry_name>" strings.

    Returns:
        A set of identifiers. If return_identifier is False, these are full
        "<DB>_<ID>_<entry_name>" strings. If True, they are just the "<ID>" portions.

    Examples:
        # Without replace or filtering
        >>> extract_and_filter_biomedical_entities("BRCA1 interacts with RAD51.")
        {"HGNC_1100_BRCA1", "HGNC_988_RAD51"}

        # With replace to handle a slash, and filtering to HGNC only
        >>> extract_and_filter_biomedical_entities(
        ...     "BRCA1/BRCA2 mutations are common.",
        ...     replace=["/"],
        ...     type_subsets=["HGNC"],
        ...     return_identifier=False
        ... )
        {"HGNC_1100_BRCA1", "HGNC_1101_BRCA2"}

        # Return only the numeric IDs (middle segment)
        >>> extract_and_filter_biomedical_entities(
        ...     "EGFR-T790M variant occurs in lung cancer.",
        ...     replace=["-"],
        ...     type_subsets=["HGNC"],
        ...     return_identifier=True
        ... )
        {"3236"}  # (assuming HGNC_3236_EGFR is the top match)

    """
    collected_entities: set[str] = set()

    base_entities = single_pass_extract_biomedical_entities(text)
    collected_entities.update(base_entities)

    if replace:
        cleaned_text = text
        for token in replace:
            cleaned_text = cleaned_text.replace(token, " ")
        cleaned_entities = single_pass_extract_biomedical_entities(cleaned_text)
        collected_entities.update(cleaned_entities)

    if type_subsets:
        filtered = {eid for eid in collected_entities if any(eid.startswith(ns) for ns in type_subsets)}
        collected_entities = filtered

    if return_identifier:
        identifier_only = {eid.split("_")[-1] for eid in collected_entities}
        return identifier_only

    return collected_entities


def single_pass_extract_biomedical_entities(text: str) -> set[str]:
    """
    Run Gilda NER and grounding on input text, returning normalized IDs.

    Gilda will scan the text for biomedical mentions (genes, proteins, chemicals,
    etc.) and link each span to a canonical entry in its ontologies (e.g., HGNC,
    MESH, GO, CHEBI, EFO). For each detected span, we take the top match
    (`matches[0]`) and build a string of the form `<db>_<id>_<entry_name>`.
    Any identifiers listed in BANNED_ENTITIES are removed before returning.

    Args:
        text: Free-form text in which to locate and normalize biomedical entities.

    Returns:
        A set of strings formatted as `<db>_<id>_<entry_name>`. Examples:
          - "HGNC_11998_TP53"
          - "MESH_D012345_Leukemia"
        Entries in BANNED_ENTITIES (e.g., generic GO terms or uninformative tags)
        will be excluded.

    Notes:
      - `gilda.annotate(text)` returns annotations, each with a `.matches` list
        sorted by confidence. We select `matches[0]` for simplicity.
      - If you need more control (e.g., filter by score), inspect `annotation.matches[0].score`.
      - BANNED_ENTITIES is a set of fully qualified IDs to drop (e.g.,
        "GO_GO:0008150_biological_process").

    Example:
        >>> single_pass_extract_biomedical_entities("TP53 regulates cell cycle.")
        {"HGNC_11998_TP53", "GO_GO:0007049_cell_cycle"}

    """
    get_stopwords()
    annotations = gilda.annotate(text)
    entities: set[str] = set()

    for ann in annotations:
        top_match = ann.matches[0].term
        normalized_id = f"{top_match.db}_{top_match.id}_{top_match.entry_name}"
        entities.add(normalized_id)

    return entities.difference(BANNED_ENTITIES)


def get_stopwords(lang: str = "english") -> None:
    """
    Get the stopwords for a given language.

    Deferred download of stopwords to only trigger at usage to avoid unnecessary downloads.
    Will also help with testing as can mock the get_entities function.

    Args:
        lang (str): The language for which to get the stopwords. Defaults to "english".

    """
    try:
        stopwords.words(lang)
    except LookupError:
        nltk.download("stopwords", quiet=True)
