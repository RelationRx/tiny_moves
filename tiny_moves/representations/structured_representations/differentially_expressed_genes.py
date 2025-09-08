import logging
from typing import Literal

from pydantic import BaseModel

from tiny_moves.metrics.utils.named_entity_recognition import extract_and_filter_biomedical_entities

logger = logging.getLogger(__name__)


DIFFERENTIALLY_EXPRESSED_DIRECTION = Literal["UP", "DOWN", "UNDEFINED"]


class DifferentiallyExpressedGene(BaseModel):
    """
    A class to hold the differentially expressed gene.

    hgnc_symbol: str
    differentially_expressed: Literal["UP", "DOWN", "UNDEFINED"] The differentially expressed status of the gene.
    If compensatory feedback mechanisms are described that give conflicting directions,
    prefer assigning the primary change reported rather than UNDEFINED,
    """

    hgnc_symbol: str
    differentially_expressed: DIFFERENTIALLY_EXPRESSED_DIRECTION


class DifferentiallyExpressedGenes(BaseModel):
    """A class to hold a list of differentially expressed genes in a hypothesis."""

    differentially_expressed_genes: list[DifferentiallyExpressedGene]

    def to_str(self) -> str:
        """Return a human-readable string representation of the differentially expressed genes."""
        return ", ".join(
            [f"{gene.hgnc_symbol}: {gene.differentially_expressed}" for gene in self.differentially_expressed_genes]
        )

    def _normalise_gene_symbols(self, gene_name: str) -> str | None:
        """
        Normalise gene symbols by extracting biomedical entities.

        Args:
            gene_name: The gene name to normalise.

        Returns:
            A single normalised gene symbol or None if multiple matches/no matches

        """
        normalised = list(
            extract_and_filter_biomedical_entities(gene_name, type_subsets=["HGNC"], return_identifier=True)
        )

        if len(normalised) > 1:
            logger.warning("Multiple gene symbols found for %s: %s", gene_name, normalised)
            return None

        if not normalised:
            # to handle false negatives in the gilda NER
            logger.warning("No gene symbol found for %s, returning input surface form", gene_name)
            return gene_name

        return normalised[0]

    def get_differentially_expressed_genes(
        self, direction: DIFFERENTIALLY_EXPRESSED_DIRECTION, return_identifier: bool = True
    ) -> list[str]:
        """Get the genes that are differentially expressed in a specific direction."""
        diff_expr_genes = [
            x.hgnc_symbol for x in self.differentially_expressed_genes if x.differentially_expressed == direction
        ]
        if return_identifier:
            return [normalised_gene for x in diff_expr_genes if (normalised_gene := self._normalise_gene_symbols(x))]
        else:
            return diff_expr_genes
