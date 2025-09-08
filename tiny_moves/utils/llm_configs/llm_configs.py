from typing import Union

from autogen.logger.base_logger import ConfigItem
from autogen.runtime_logging import LLMConfig
from pydantic import BaseModel

from tiny_moves.experimental.corruption.generate_dataset.structured_outputs import ListOfCorruptions
from tiny_moves.metrics.pairwise_hypothesis_metrics.score_errors.structured_outputs import (
    ListOfStatementPairEvaluations,
)
from tiny_moves.representations.answers.directionality import Directionality
from tiny_moves.representations.structured_representations.differentially_expressed_genes import (
    DifferentiallyExpressedGenes,
)

from tiny_moves.representations.structured_representations.hypothesis import (
    Hypotheses,
    Hypothesis,
)
from tiny_moves.representations.structured_representations.QA import (
    BinaryAnswer,
    BinaryAnswerWithConfidence,
    TernaryAnswerWithEvidence,
)
from tiny_moves.representations.structured_representations.triples import (
    ListOfEvaluationResults,
    ListOfTriples,
)
from tiny_moves.settings import OPENAI_API_KEY

from ._registry import register_llm_config


def make_config_list(model: str, openai_api_key: str = OPENAI_API_KEY) -> list[dict[str, str]]:
    """Make a model config dict for ag2."""
    return [{"api_key": openai_api_key, "model": model}]


StructuredLLmConfig = dict[
    str,
    Union[
        None,
        float,
        int,
        type[BaseModel],
        ConfigItem,
        list[ConfigItem],
        type[
            Hypothesis
            | BinaryAnswer
            | BinaryAnswerWithConfidence
            | Hypotheses
            | DifferentiallyExpressedGenes
            | Directionality
        ],
    ],
]


@register_llm_config()
def chatgpt_4o_config() -> LLMConfig:
    """Return a configuration for ChatGPT-4o."""
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="chatgpt-4o-latest"),
        "timeout": 540000,
    }


@register_llm_config()
def gpt4o_config() -> LLMConfig:
    """Return a configuration for GPT-4o."""
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
    }


@register_llm_config()
def gpt4o_structured_config() -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    print("NB tech debt: structured output is tied explicitly to Hypotheses class.")
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": Hypothesis,
    }


@register_llm_config()
def gpt4o_directionality_config() -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    print("NB tech debt: structured output is tied explicitly to Hypotheses class.")
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": Directionality,
    }


@register_llm_config()
def gpt4o_hypotheses_extractor_config() -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    print("NB tech debt: structured output is tied explicitly to Hypotheses class.")
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": Hypotheses,
    }


def gpt4o_ternary_evaluation(seed: int) -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    return {
        "cache_seed": seed,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": TernaryAnswerWithEvidence,
    }



@register_llm_config()
def gpt4o_differential_expression_config() -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    print("NB tech debt: structured output is tied explicitly to Hypotheses class.")
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": DifferentiallyExpressedGenes,
    }


@register_llm_config()
def gpt4o_binary_QA_config() -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    print("NB tech debt: structured output is tied explicitly to Hypotheses class.")
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": BinaryAnswer,
    }


@register_llm_config()
def gpt4o_binary_QA_w_conf_config() -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": BinaryAnswerWithConfidence,
    }


@register_llm_config()
def gpt4o_mini_config() -> LLMConfig:
    """Return a configuration for GPT-4o Mini."""
    return {
        "cache_seed": 42,  # change the cache_seed for different trials
        "temperature": 0.01,
        "config_list": make_config_list(model="gpt-4o-mini"),
        "timeout": 540000,
    }


@register_llm_config()
def gpt4o_list_of_evaluations(seed: int) -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    return {
        "cache_seed": seed,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": ListOfEvaluationResults,
    }


@register_llm_config()
def gpt4o_list_of_triples(seed: int) -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    return {
        "cache_seed": seed,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": ListOfTriples,
    }


@register_llm_config()
def gpt4o_list_of_statement_evaluations(seed: int) -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    return {
        "cache_seed": seed,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": ListOfStatementPairEvaluations,
    }


@register_llm_config()
def chatgpt4o_generate_corruptions(seed: int) -> LLMConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    return {
        "cache_seed": 22,  # change the cache_seed for different trials
        "temperature": 1,
        "config_list": make_config_list(model="chatgpt-4o-latest"),
        "timeout": 540000,
    }


@register_llm_config()
def gpt4o_structure_corruptions(seed: int) -> StructuredLLmConfig:
    """Return a configuration for GPT-4o that returns a structured output."""
    return {
        "cache_seed": 22,  # change the cache_seed for different trials
        "temperature": 0,
        "config_list": make_config_list(model="gpt-4o"),
        "timeout": 540000,
        "response_format": ListOfCorruptions,
    }
