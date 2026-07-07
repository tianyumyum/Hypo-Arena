---
name: diagnostic_reasoning
description: >
  Diagnostic Reasoning. Identifies which evidence items have the highest
  diagnostic value — i.e., which facts most effectively discriminate between
  competing explanations. Best suited for cases with abundant evidence where
  the challenge is finding the key discriminating facts amid noise.
---

## Analytical Framework: Diagnostic Reasoning

Apply diagnostic reasoning to generate hypotheses from the source context.
This method prioritizes evidence quality over quantity — a single highly
diagnostic fact outweighs ten pieces of ambiguous supporting evidence.

### Core Concept

**Diagnostic evidence** is evidence whose presence or absence would be
substantially different under one hypothesis compared to another. Non-diagnostic
evidence is consistent with multiple hypotheses and therefore cannot help
distinguish between them.

Example:
- "The system showed anomalous behavior" is non-diagnostic — it is consistent
  with many different explanations.
- "The pressure reading exceeded the rated limit by 300% while the relief
  valve remained closed" is highly diagnostic — it strongly favors a blocked
  safety device over a gradual process deviation.

### Procedure

1. **Survey the evidence landscape**: Read through the entire context and
   catalog the key facts: observations, measurements, conditions, actions,
   temporal patterns, and documented findings.

2. **Generate initial hypothesis candidates**: Based on domain knowledge,
   identify 3-5 plausible explanations for the observed outcome or pattern.

3. **Score evidence diagnosticity**: For each major fact, ask:
   - "Would I expect to see this fact if Hypothesis A is true?"
   - "Would I expect to see this fact if Hypothesis B is true?"
   - If the answer differs significantly between hypotheses, the evidence
     is diagnostic. If the answer is similar, the evidence is non-diagnostic.

4. **Rank evidence by diagnostic power**: Order evidence from most to least
   diagnostic. The most diagnostic evidence should drive hypothesis selection.

5. **Build hypotheses from diagnostic evidence outward**: Start with the
   2-3 most diagnostic facts and construct hypotheses that best explain
   this critical evidence. Then check consistency with remaining evidence.

6. **Flag evidence gaps**: Identify diagnostic questions that the available
   evidence does NOT answer. These gaps represent analytical uncertainties
   and should be reflected in the hypothesis framing.

### Output Mapping

Translate the methodology's analytical output into the Hypothesis–Evidence
pairs your domain instructions specify; cardinality, category labels
(where applicable), and Evidence-anchoring rules are defined there. Use
the methodology's outputs as analytical inputs, not as output structure
— do not surface internal scaffolding (matrices, scoring tables,
step-by-step procedure, dimension lists) in the final output.

### Quality Criteria

- Every hypothesis must be anchored to at least one piece of high-diagnosticity
  evidence, explicitly named in the verification.
- Verification should explain WHY the cited evidence is diagnostic — what
  alternative explanation it rules out.
- Avoid building hypotheses primarily from non-diagnostic evidence (general
  summaries, background conditions) without diagnostic anchoring.
