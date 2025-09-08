import json

from autogen import AssistantAgent

from tiny_moves.metrics.pairwise_hypothesis_metrics.score_errors.structured_outputs import (
    ListOfStatementPairEvaluations,
    ListOfStatementPairs,
)
from tiny_moves.metrics.pairwise_hypothesis_metrics.score_errors.utils import (
    format_statement_pairs_as_prompt,
)
from tiny_moves.utils.llm_configs.llm_configs import chatgpt_4o_config, gpt4o_list_of_statement_evaluations

EVALUATOR_PROMPT = """
You are an evaluator of biological pathways.

You are given pairs of statements: (correct statement, corrupted statement)
You are also given a candidate biological pathway.

The difference between the corrupted and correct statement is an error introduced by a corruption operation.

Your task:
We are evaluating the error persistence score.
For each pair of correct-corrupted statements:

1. Return 1 if the error introduced by the corrupted statement is present in the candidate pathway
2. Return 0 otherwise.

You may encounter the following errors:
correct: A phosophylates B
corrupted: A phosphorylates C
The error is the incorrect entity C.

correct: A phosophorylates B
corrupted: A dephosphorylates B
The error is the incorrect relationship

You may also encounter cases where a new statement, which is hallucinated or completely irrelevant is added.
In that case, the correct statement will be blank and the corrupted statement will be the addition.
Your job is then to check whether the hallucination / irrelevant statement is present.
If it is removed completely or correctly connected to the candidate pathway, return 0.


Return your answer in the following format:

correct: str
corrupted: str
relevant_fragment_from_candidate: str
score: float
reason_for_score: str

"""


def evaluate_presence_of_errors_in_candidate(
    seed: int,
    list_of_statement_pairs: ListOfStatementPairs,
    candidate_text: str,
) -> ListOfStatementPairEvaluations:
    """
    Evaluate the presence of errors in a candidate text against a list of statement pairs.

    Args:
        seed: An integer seed for reproducibility.
        list_of_statement_pairs: A ListOfStatementPairs object containing pairs of correct and erroneous statements.
        candidate_text: A string representing the candidate text to evaluate.

    Returns:
        ListOfStatementPairEvaluations: A structured output containing the evaluation results for each statement pair.

    """
    evaluator = AssistantAgent(
        name="evaluator",
        system_message=EVALUATOR_PROMPT,
        llm_config=chatgpt_4o_config(),
    )

    structuror = AssistantAgent(
        name="structuror",
        system_message="You are a structured output parser. Your job is to structure data.",
        llm_config=gpt4o_list_of_statement_evaluations(seed=seed),
    )

    user_prompt = f"""
        Pairs of statements:
        \"\"\"
        {format_statement_pairs_as_prompt(list_of_statement_pairs.pairs)}
        \"\"\"

        Pathway to evaluate: :
        \"\"\"
        {candidate_text}
        \"\"\"
        """

    evaluator_reply = evaluator.generate_reply([{"role": "user", "content": user_prompt}])
    evaluator_content = evaluator_reply["content"] if isinstance(evaluator_reply, dict) else evaluator_reply

    structured_reply = structuror.generate_reply([{"role": "user", "content": evaluator_content}])
    structured_content = structured_reply["content"] if isinstance(structured_reply, dict) else structured_reply
    parsed = json.loads(structured_content)
    return ListOfStatementPairEvaluations.model_validate(parsed)
