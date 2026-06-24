def calculate_jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Calculates the Jaccard Similarity between two sets of shingles.
    Formula: J(A,B) = |A intersection B| / |A union B|
    Returns a float between 0.0 (no similarity) and 1.0 (identical).
    """
    # Calculate intersection and union sizes
    intersection_size = len(set_a.intersection(set_b))
    union_size = len(set_a.union(set_b))

    # Avoid division by zero
    if union_size == 0:
        return 0.0

    return intersection_size / union_size