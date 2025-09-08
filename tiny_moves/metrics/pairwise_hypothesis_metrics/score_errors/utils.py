from tiny_moves.metrics.pairwise_hypothesis_metrics.score_errors.structured_outputs import StatementPair


def format_statement_pairs_as_prompt(
    pairs: list[StatementPair],
) -> str:
    """
    Format a list of statement pairs into a string representation suitable for prompts.

    Args:
        pairs: A list of StatementPair objects to format.

    Returns:
        A string representation of the statement pairs, formatted as "correct statement - corrupted statement".

    """
    lines = []
    for pair in pairs:
        lines.append(f"correct: {pair.correct} - corrupted: {pair.corrupted}")
    return "\n".join(lines)
