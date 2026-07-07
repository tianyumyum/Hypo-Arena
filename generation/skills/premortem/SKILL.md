---
name: premortem
description: >
  Premortem Analysis. Assumes the most likely initial hypothesis is WRONG,
  then works backward to identify where the reasoning could fail. Best suited
  for cases where there is a strong initial conclusion or consensus that
  deserves adversarial challenge before acceptance.
---

## Analytical Framework: Premortem Analysis

Apply the Premortem method to generate hypotheses from the case context.
This method combats premature closure — the tendency to lock onto the first
plausible explanation and stop looking for alternatives.

### Core Concept

In a standard analysis, you gather evidence and build toward a conclusion.
In a premortem, you START with the conclusion and imagine it has been proven
wrong, then reason backward to identify how and why it could fail.

This is the analytical equivalent of "red-teaming" your own hypothesis.

### Procedure

1. **Form the leading hypothesis**: Read the context and identify the most
   obvious, most intuitive explanation for the case. This is your
   "initial best guess" — the explanation that would most likely be offered
   in a first-pass analysis.

2. **Declare it wrong**: Explicitly assume: "This hypothesis has been
   proven incorrect. The actual cause was something else." Accept this
   as a given for the next steps.

3. **Backward reasoning — where could we be wrong?**
   For each element of the leading hypothesis, ask:
   - **Misread evidence**: "Which facts might we be interpreting incorrectly?
     Could a measurement be wrong? Could a reported action not have
     happened as described?"
   - **Missing evidence**: "What information is absent from the context
     that we are implicitly assuming doesn't matter? What if it does?"
   - **Alternative causal paths**: "What other failure chain could produce
     the same observable evidence? What if the timeline sequence is
     correlation, not causation?"
   - **Overlooked actors/factors**: "Are there system components, human
     actors, environmental factors, or organizational pressures that we
     haven't considered?"

4. **Construct alternative hypotheses**: Based on the backward reasoning,
   build 1-3 alternative hypotheses that:
   - Explain the same evidence through a different causal mechanism
   - Account for evidence that the leading hypothesis struggles to explain
   - Address the specific vulnerabilities identified in step 3

5. **Synthesize**: Combine insights from the leading hypothesis and its
   alternatives. The final output may include:
   - A refined version of the original hypothesis (if the premortem
     revealed it was essentially correct but needed nuance)
   - A genuinely different hypothesis (if the premortem revealed a
     fundamental flaw in the original reasoning)
   - A hybrid hypothesis (if the premortem revealed that the truth
     likely involves elements of multiple explanations)

### Output Mapping

Translate the methodology's analytical output into the Hypothesis–Evidence
pairs your domain instructions specify; cardinality, category labels
(where applicable), and Evidence-anchoring rules are defined there. Use
the methodology's outputs as analytical inputs, not as output structure
— do not surface internal scaffolding (matrices, scoring tables,
step-by-step procedure, dimension lists) in the final output.

### Quality Criteria

- The leading hypothesis must be explicitly stated before being
  challenged. The reader should see the full premortem arc.
- Verification must reference specific weaknesses found in the
  leading hypothesis during the backward reasoning step.
- Alternative hypotheses must be genuinely different from the leading
  hypothesis, not minor variations.
- Avoid strawman challenges — the premortem should target real
  vulnerabilities in the reasoning, not trivially unlikely scenarios.
