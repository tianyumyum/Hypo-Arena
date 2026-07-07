---
name: assumptions
description: >
  Key Assumptions Check. Systematically identifies and stress-tests the
  implicit assumptions underlying an analysis. Best suited for cases that
  appear straightforward but may rest on unexamined premises that, if wrong,
  would fundamentally change the conclusion.
---

## Analytical Framework: Key Assumptions Check

Apply the Key Assumptions Check to generate hypotheses from the case
context. This method targets the hidden premises that analysts unconsciously
rely on — assumptions that, if incorrect, would invalidate the entire
analytical chain.

### Core Concept

Every analysis rests on assumptions — beliefs treated as facts without
explicit verification. These include:
- **Causal assumptions**: "X caused Y" (but was X actually sufficient? Were
  there confounders?)
- **State assumptions**: "The system was in state S" (but was the sensor
  reading accurate? Was the state as reported?)
- **Behavioral assumptions**: "The operator did/knew/intended Z" (but what
  evidence actually confirms this?)
- **Absence assumptions**: "Factor F was not present" (but was it actually
  checked, or just not mentioned?)

A **key assumption** is one where: (1) the analysis depends on it being true,
AND (2) there is reason to question it.

### Procedure

1. **Form an initial assessment**: Based on a first reading of the context,
   what is the most obvious explanation for the case? Write it down.

2. **Surface implicit assumptions**: Systematically list every assumption
   embedded in that initial assessment. For each factual claim in your
   reasoning chain, ask:
   - "What am I assuming to be true here?"
   - "Is this directly evidenced or am I inferring it?"
   - "Could this be wrong, incomplete, or misleading?"

3. **Categorize assumptions by vulnerability**:
   - **Well-supported**: Direct evidence in the context confirms this.
   - **Reasonable but unverified**: Plausible but no direct evidence.
   - **Potentially fragile**: There are hints in the context that this
     might not hold (e.g., conflicting data, unusual circumstances,
     missing information).

4. **Identify linchpin assumptions**: Which assumptions, if wrong, would
   cause the entire analysis to collapse? These are the linchpins.

5. **Generate alternative hypotheses from fragile assumptions**: For each
   fragile or linchpin assumption, construct an alternative hypothesis
   that assumes the opposite. Ask: "If this assumption is false, what
   would the case look like instead?"

6. **Assess the alternative**: Does the alternative hypothesis explain the
   evidence as well or better than the original? Are there facts that only
   the alternative can explain?

### Output Mapping

Translate the methodology's analytical output into the Hypothesis–Evidence
pairs your domain instructions specify; cardinality, category labels
(where applicable), and Evidence-anchoring rules are defined there. Use
the methodology's outputs as analytical inputs, not as output structure
— do not surface internal scaffolding (matrices, scoring tables,
step-by-step procedure, dimension lists) in the final output.

### Quality Criteria

- Each hypothesis must explicitly name the assumption being challenged.
- Verification must explain both why the assumption is fragile AND how
  the alternative hypothesis accounts for the available evidence.
- Avoid challenging assumptions that are firmly established by physical
  evidence — focus on assumptions about processes, states, behaviors,
  and causal links that are inferred rather than directly observed.
