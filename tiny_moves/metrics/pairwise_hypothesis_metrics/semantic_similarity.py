from typing import Literal

from sacrebleu import BLEU, CHRF, TER
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tiny_moves.metrics.core_properties.similarity import pairwise_jaccard
from tiny_moves.metrics.pairwise_hypothesis_metrics.template import (
    PairwiseHypothesisMetric,
)
from tiny_moves.metrics.utils.embed import embed_text_with_sentence_transformer
from tiny_moves.metrics.utils.named_entity_recognition import (
    extract_and_filter_biomedical_entities,
)
from tiny_moves.representations.template import ProtocolStr


class JaccardSimilarity(PairwiseHypothesisMetric[ProtocolStr, float]):
    """A metric that computes the Jaccard similarity between two hypotheses based on their entities."""

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
        type_subsets: list[Literal["HGNC", "MESH", "CHEBI", "GO", "EFO"]] | None = None,
    ) -> float:
        """
        Compute the Jaccard similarity between two hypotheses based on their entities.

        Note: high confidence types are HGNC and GO.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.
            type_subsets: Optional list of entity type subsets to filter the entities.

        Returns:
            float: The Jaccard similarity score between the two hypotheses.

        """
        reference_hypothesis_str = reference_hypothesis.to_str()
        candidate_hypothesis_str = candidate_hypothesis.to_str()
        reference_entities = extract_and_filter_biomedical_entities(reference_hypothesis_str, type_subsets=type_subsets)
        candidate_entities = extract_and_filter_biomedical_entities(candidate_hypothesis_str, type_subsets=type_subsets)
        jaccard_index = pairwise_jaccard(reference_entities, candidate_entities)
        return jaccard_index


class BLEUScore(PairwiseHypothesisMetric[ProtocolStr, float]):
    """Compute the BLEU score between two hypotheses."""

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
        smooth_method: str = "exp",  # Options: "floor", "add-k", "exp", "none"
    ) -> float:
        """
        Compute the BLEU score between two hypotheses.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.
            smooth_method: The smoothing method to use for BLEU score calculation.

        Returns:
            float: The BLEU score between the two hypotheses.

        """
        reference_hypothesis_str = reference_hypothesis.to_str()
        candidate_hypothesis_str = candidate_hypothesis.to_str()

        metric = BLEU(smooth_method=smooth_method)
        score = metric.corpus_score(
            [candidate_hypothesis_str],
            [[reference_hypothesis_str]],  # sacrebleu expects a list of references for each candidate
        )

        out: float = score.score

        return out


class CHRFScore(PairwiseHypothesisMetric[ProtocolStr, float]):
    """
    Compute the CHRF score between two hypotheses.

    It is a character n-gram F-score metric that computes the similarity based on character n-grams.
    It ranges from 0 to 100, where 100 means a perfect match.

    chrF = 2 * (Precision * Recall) / (Precision + Recall)
    """

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
    ) -> float:
        """
        Compute the CHRF score between two hypotheses.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.

        Returns:
            float: The CHRF score between the two hypotheses. Range is from 0-100.

        """
        reference_hypothesis_str = reference_hypothesis.to_str()
        candidate_hypothesis_str = candidate_hypothesis.to_str()

        metric = CHRF()
        score = metric.corpus_score(
            [candidate_hypothesis_str],
            [[reference_hypothesis_str]],  # sacrebleu expects a list of references for each candidate
        )

        out: float = score.score

        return out


class TERScore(PairwiseHypothesisMetric[ProtocolStr, float]):
    """
    Compute the TER score between two hypotheses.

    TER (Translation Edit Rate) is a metric that measures the number of edits
    required to change a candidate translation into a reference translation.

    It ranges from 0 to inf, where 0 means no edits are needed (perfect match).

    TER = Number of edits / Average reference length
    """

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
    ) -> float:
        """
        Compute the TER score between two hypotheses.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.

        Returns:
            float: The TER score between the two hypotheses. Range is from 0-100.

        """
        reference_hypothesis_str = reference_hypothesis.to_str()
        candidate_hypothesis_str = candidate_hypothesis.to_str()

        metric = TER()
        score = metric.corpus_score(
            [candidate_hypothesis_str],
            [[reference_hypothesis_str]],  # sacrebleu expects a list of references for each candidate
        )

        out: float = score.score

        return out


class SemanticSimilarityViaTFIDF(PairwiseHypothesisMetric[ProtocolStr, float]):
    """Compute semantic similarity between two hypotheses using TF-IDF vectorization and cosine similarity."""

    def __init__(self) -> None:
        """
        Initialize the SemanticSimilarityViaTFIDF metric with a TF-IDF vectorizer.

        This vectorizer will be used to transform the hypotheses into TF-IDF vectors for similarity computation.
        """
        super().__init__()
        self.vectorizer = TfidfVectorizer()

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
    ) -> float:
        """
        Compute the semantic similarity between two hypotheses using TF-IDF vectorization and cosine similarity.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.

        Returns:
            float: The cosine similarity score between the two hypotheses.

        """
        reference_hypothesis_str = reference_hypothesis.to_str()
        candidate_hypothesis_str = candidate_hypothesis.to_str()
        self.vectorizer.fit([reference_hypothesis_str, candidate_hypothesis_str])
        vectorized_reference = self.vectorizer.transform([reference_hypothesis_str])
        vectorized_candidate = self.vectorizer.transform([candidate_hypothesis_str])
        similarity = cosine_similarity(vectorized_reference, vectorized_candidate)
        return float(similarity[0][0])


class SemanticSimilarityViaSentenceEmbedding(PairwiseHypothesisMetric[ProtocolStr, float]):
    """Compute semantic similarity between two hypotheses using sentence embeddings and cosine similarity."""

    def __init__(self, model_tag: str = "NeuML/pubmedbert-base-embeddings") -> None:
        """
        Initialize the SemanticSimilarityViaSentenceEmbedding metric.

        This metric uses sentence embeddings to compute semantic similarity between hypotheses.
        """
        super().__init__()
        self.model = SentenceTransformer(model_tag)

    def compute(
        self,
        reference_hypothesis: ProtocolStr,
        candidate_hypothesis: ProtocolStr,
    ) -> float:
        """
        Compute the semantic similarity between two hypotheses using sentence embeddings and cosine similarity.

        Args:
            reference_hypothesis: The reference hypothesis.
            candidate_hypothesis: The candidate hypothesis.
            model_tag: The model tag for the SentenceTransformer.

        Returns:
            float: The cosine similarity score between the two hypotheses.

        """
        reference_hypothesis_str = reference_hypothesis.to_str()
        candidate_hypothesis_str = candidate_hypothesis.to_str()
        emb1 = embed_text_with_sentence_transformer(reference_hypothesis_str, self.model)
        emb2 = embed_text_with_sentence_transformer(candidate_hypothesis_str, self.model)
        emb1, emb2 = emb1.reshape(1, -1), emb2.reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)
        return float(similarity[0][0])
