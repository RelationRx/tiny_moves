def pairwise_jaccard(bag_1: set[str], bag_2: set[str]) -> float:
    """
    Calculate the pairwise Jaccard index for a pair of bag of entities.

    Higher values indicate more similarity between the two bags of entities.
    The range is [0.0, 1.0], where 1.0 means the bags are identical.

    Take entity-tagged output from get_hypothesis_entities.

    Args:
        bag_1 (set): The first bag of entities.
        bag_2 (set): The second bag of entities.

    Returns:
        list: The Jaccard index values for all pairs of hypotheses.

    """
    if len(bag_1.intersection(bag_2)) > 0:
        jacc = len(bag_1.intersection(bag_2)) / len(bag_1.union(bag_2))
    else:
        jacc = 0.0

    return jacc
