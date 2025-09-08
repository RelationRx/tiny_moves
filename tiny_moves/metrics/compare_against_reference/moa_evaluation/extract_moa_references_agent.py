import json

from autogen import AssistantAgent

from tiny_moves.representations.structured_representations.triples import PREDICATES, ListOfTriples
from tiny_moves.utils.llm_configs.llm_configs import gpt4o_list_of_triples

EXTRACT_MOA_REFERENCE_STATEMENTS_PROMPT = f"""
Extract Unique Mechanistic Pathway Steps.
Your task is to extract unique, mechanistic, directional pathway steps from biomedical text. Each step must describe a
causal biological relationship between molecules, genes, proteins, complexes, pathways, or biological processes.

🧾 Format
Output each step as:
[Source entity] → [predicate] → [Target entity]
Use only these predicates to label the relationships: {PREDICATES}

✅ Rules (Critical)
Do not include any causal step more than once, even if:

It's phrased differently
It's implied in multiple places
It's expressed using a different relation verb or noun phrasing

Before adding a step, check against the list of already added steps. If the same logic appears in a new form, skip it.
Prefer mechanistically detailed steps. Avoid causal shortcuts if they exist . e.g., prefer “X promotes Z, Z promotes Y”
over simply “X promotes Y”.
Only include "X promotes Y" if it is explicitly stated in the text.
Use consistent terminology. Choose a canonical form for each entity (e.g., COL1A1)
and use it throughout. If multiple variants appear, pick one and discard the rest.


🧪 Example
Input:
“Gene X activates Pathway Y. Pathway Y affects transcription of Gene Z. Gene X also impairs Protein A activity,
which upregulates Gene Z. Protein A forms a complex with Protein B.”

Output:
Gene X → activates → Pathway Y
Pathway Y → regulates → transcription of Gene Z
Gene X → inhibits → Protein A
Protein A → upregulates → Gene Z
Protein B → binds to → Protein B

(No duplication. All steps are mechanistic. No vague relations.)


📍 Task
Extract all unique, mechanistic steps from the following input. Maintain a global list of previously emitted steps
and ensure no causal step is duplicated.

"""


def extract_pathway_triples(reference_text: str, seed: int = 42) -> ListOfTriples:
    """
    Extract mechanistic pathway steps from a given hypothesis text using a structured LLM agent.

    Args:
        reference_text: The input text containing the hypothesis from which to extract mechanistic steps.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        List of tuples, each containing a source entity, relation/action, and target entity.

    """
    agent = AssistantAgent(
        name="triple_extractor",
        system_message=EXTRACT_MOA_REFERENCE_STATEMENTS_PROMPT,
        llm_config=gpt4o_list_of_triples(seed=seed),
    )

    user_prompt = f"""
        Input text:
        \"\"\"
        {reference_text}
        \"\"\"
        """

    reply = agent.generate_reply([{"role": "user", "content": user_prompt}])
    content = reply["content"] if isinstance(reply, dict) else reply
    parsed = json.loads(content)
    return ListOfTriples.model_validate(parsed)
