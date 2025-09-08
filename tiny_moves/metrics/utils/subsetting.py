from typing import Literal

import pandas as pd

from tiny_moves.metrics.utils.named_entity_recognition import (
    extract_and_filter_biomedical_entities,
)


def subset_dataframe_by_entities(
    data_df: pd.DataFrame,
    id_col: str,
    query_string: str,
    ner_entity_type: Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"],
) -> pd.DataFrame:
    """
    Subset a DataFrame to only those entities in a query string.

    Note: high confidence types are HGNC and GO.

    Args:
        data_df: The DataFrame to subset.
        id_col: The column in `data_df` that contains the entity identifiers.
        query_string: The string to match entities against.
        ner_entity_type: The type of entities to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows where the entity identifier is in the query string.

    """
    entities = extract_and_filter_biomedical_entities(
        query_string, replace=["-", "/"], type_subsets=[ner_entity_type], return_identifier=True
    )

    allowed = list(entities)
    return data_df.loc[data_df[id_col].isin(allowed)]
