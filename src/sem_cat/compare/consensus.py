"""Consensus clustering across multiple model outputs."""

from __future__ import annotations

from src.sem_cat.compare.data_structures import ConsensusCluster, ModelOutput
from src.sem_cat.compare.normalization import output_similarity


def cluster_outputs(
    outputs: list[ModelOutput],
    threshold: float = 0.85,
) -> list[ConsensusCluster]:
    """Greedy clustering of model outputs by similarity.

    Args:
        outputs: List of non-blank model outputs with normalized_gloss_en set.
        threshold: Similarity threshold for clustering.

    Returns:
        List of ConsensusCluster objects, sorted by cluster size descending.
    """
    clusters: list[ConsensusCluster] = []

    for output in outputs:
        if not output.normalized_gloss_en:
            continue

        placed = False
        for cluster in clusters:
            if output_similarity(
                output.normalized_gloss_en,
                cluster.representative,
            ) >= threshold:
                cluster.model_keys.append(output.model_key)
                cluster.outputs.append(output)
                placed = True
                break

        if not placed:
            clusters.append(ConsensusCluster(
                representative=output.normalized_gloss_en,
                model_keys=[output.model_key],
                outputs=[output],
            ))

    # Sort by cluster size descending
    clusters.sort(key=lambda c: len(c.model_keys), reverse=True)
    return clusters
