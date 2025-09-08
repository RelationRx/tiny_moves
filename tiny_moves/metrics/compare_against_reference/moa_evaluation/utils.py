from tiny_moves.representations.structured_representations.triples import SignedTriple


def format_triples_as_prompt(signed_triples: list[SignedTriple]) -> str:
    """
    Format a list of triples into a string representation suitable for prompts.

    Args:
        signed_triples: A list of SignedTriple objects to format.

    Returns:
        A string representation of the triples, formatted as "subject → predicate → object".

    """
    lines = []
    for signed_triple in signed_triples:
        lines.append(
            f"{signed_triple.triple.subject} → {signed_triple.triple.predicate} → {signed_triple.triple.object}. "
            f"Directional predicate: {signed_triple.directional_predicate}"
            f"Predicate sign: {signed_triple.predicate_sign}"
        )
    return "\n".join(lines)
