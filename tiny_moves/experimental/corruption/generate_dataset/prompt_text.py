SHARED_INSTRUCTIONS = """
You are a biomedical expert tasked with introducing a specific type of error into biological pathways.

You will be given:
- a pathway title (string)
- a list of pathway steps

Your task:
- iterate through every step in the pathway
- for each step, generate a SINGLE error for EVERY difficulty level (1 and 2) of the corruption type provided below.

Return ONLY a JSON object matching this schema:
{
  "corruptions": [
    {
      "anchor_step_index": <int>,
      "operation": "replace" | "insert_before" | "insert_after",
      "error_type": "<error_type_here>",
      "difficulty": 1 | 2,
      "original_statement": <string or null>,
      "corrupted_statement": <string>,
      "category_rationale": <string>
    },
    ...
  ]
}

Constraints:
- Only the provided indices (0-based) in the steps list can be used for anchor_step_index.
- Produce mechanistic pathway steps, not measurements or assay descriptions.
  Disallow phrases like "activity is measured", "assay", "substrate used in photosynthesis",
  "novel coactivator" etc., unless the pathway already includes it.
- Respect organism/system context. Disallow entities/organelles not present in the implied system
(e.g., chlorophyll/chloroplasts in mammals; bacterial-only cofactors in human pathways) unless the pathway explicitly
includes them.
- For "replace": original_statement MUST exactly match the step at anchor_step_index.
- For inserts: original_statement MUST be null.
- Ensure variety and AVOID DUPLICATES AT ALL COST!! DO NOT INTRODUCE THE SAME ENTITY OR ERROR TYPE MULTIPLE TIMES.
- Keep each statement short (1 sentence).
- Output only valid JSON.
- in all caases, the original statement must be VERBATIM exactly as the original statement,
unless the error is "insert after", in which case you leave it blank.

The following terms are FORBIDDEN - do NOT use them in any context:
chlorophyll, chloroplast, insulin, hemoglobin, “novel coactivator”, “activity is measured”

This is the error you need to introduce:
"""

WRONG_ENTITY = """
## Wrong Entity

### What it is
Uses the wrong biological entity (gene/protein/complex/isoform/state/species) for an otherwise valid step.
The verb/edge remains unchanged, but one of the entities is changes to something that is clearly wrong.

### What it tests
Entity grounding & role appropriateness under constraints.

## Allowed Operation: Replace i.e. take an existing statement and replace the entity with a wrong one.
DO NOT change the relationship between entities, ONLY the entity itself.
Change one entity only; keep verb, product, and polarity identical.

Do not add extra entities or agents to the statement!!!!
Change the entity only — never append new ones

### Example of Allowed Corruption
original statement 1: A scavenges B
corrupted statement 1: A scavenges X

do NOT perform corruptions like this:
original statement 2: A gets cleaved to B
corrupted statement 2: A gets cleaved to B by X <- this introduces new entity X, which is not an allowed corruption

### Easy (1)
Obvious domain/type mismatch or explicit species misuse.
Keep the change simple and obvious, avoiding nuanced or subtle changes that require deep biological knowledge to detect.
The mismatch should be immediately clear to someone with moderate domain familiarity.
Swap to a correct-type-but-wrong actor (e.g., an enzyme that could catalyze something, but not this reaction).
Avoid cofactors/organelles that are wildly out-of-domain (reserve those for trivial sanity checks only sparingly).

### Hard (2)
Paralog/isoform or complex↔subunit or closely related cofactors swaps where both are plausible;
PTM/state gating omitted but required. Do not swap for non-existent entities.

### Avoid
Deep structural evidence, tumor-subtype rewiring, multi-paper ortholog/isoform disambiguation.


### NOTE:
- Make sure that the resulting erroneous statment is truly and univocally erroneous
i.e., it could never exist in this pathway, even if the order of steps was different.
- Do not swap to an entity of the same general class or role that only differs in fine-grained detail
(e.g., early vs late endosome)
- Keep the changes diverse - pick truly broadly across the spectrum of errors.
"""

WRONG_DIRECTION = """
## Wrong Direction / Wrong Relationship

### What it is
Correct entities, wrong relationship subject↔object, activate↔inhibit, upstream↔downstream.

### What it tests
Causal structure, role semantics, relationships:
who acts on whom; is the sign/order consistent with known cascade fragments; what is the correct way
in which these entities interact?

## Allowed Operation: Replace i.e. take an existing statement and replace the relationship between the entities
with a wrong one. DO NOT change the entities, ONLY the relationship itself.
Keep all entities and products identical; only change the verb between them.
If the relationship between the entities is symmetric e.g. binding,
you MUST change the verb to a *directional* one:
recruits, promotes, inhibits, ubiquitinates, phosphorylates, deubiquitinates, activates, represses.


### Example
original statement 1: A binds B
corrupted statement 1: A inhibits B
original statement 2: A is reduced to B by C
corrupted statement 2: A is oxidized to B by C

### Easy (1)
Flip of a textbook interaction, or simple subject/object swap.

### Hard (2)
Inverts upstream/downstream inside a complex, or changes effect with a missing/wrong modifier.

### Avoid
Feedback/time-scale illusions; genuinely bidirectional edges; quantitative model dependencies.
"""

ADD_UNSUPPORTED_STEP = """
## Add Unsupported Step

### What it is
Inserts a new step that does not belong.
- Easy: Irrelevant = real biology but no causal bridge to the goal.
- Hard: Relevant but fabricated = plausible but non-existent edge.

### What it tests
Step existence & relevance.

## Allowed Operation: Insert After i.e. take an existing statement and introduce a new step after it.
The new step must not be supported by the current pathway or literature.

### Example
Easy: Insert “insulin receptor binding kinetics” into beta-catenin destruction steps.
Hard: “CK1alpha ubiquitinates beta-catenin” (fabricated chemistry/role).

### Easy (1)
Clearly off-path module or assay artefact.
Real biology but off-path and mechanistically disconnected (no effect if removed).
Does not touch any of the entities involved in the pathway.

### Hard (2)
Plausible, on-topic edge that contradicts known constraints or lacks curated support.
As in, it should include one or more entities in the pathway, but the step itself could not be correct
in any context or circumstances.

### Avoid
Contested proposals needing expert triage; subtle violations requiring specialized topology data.
"""

# Final prompts
WRONG_ENTITY_PROMPT = SHARED_INSTRUCTIONS + WRONG_ENTITY
WRONG_DIRECTION_PROMPT = SHARED_INSTRUCTIONS + WRONG_DIRECTION
ADD_UNSUPPORTED_STEP_PROMPT = SHARED_INSTRUCTIONS + ADD_UNSUPPORTED_STEP
