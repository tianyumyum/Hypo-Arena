---
name: ach
description: >
  Analysis of Competing Hypotheses (ACH). Systematically evaluates multiple
  competing hypotheses by constructing an explicit evidence-diagnosticity
  matrix with C/I/NA scoring. Ranks hypotheses by inconsistency score —
  the best hypothesis is the one with the least disconfirming evidence.
  Best suited for cases where evidence is contradictory, supports multiple
  plausible explanations, or where confirmation bias is a risk.
---

## Analytical Framework: Analysis of Competing Hypotheses

Apply the ACH methodology to generate hypotheses from the case context.
This method guards against confirmation bias by forcing systematic comparison
of all plausible explanations against available evidence through an explicit
evidence-diagnosticity matrix.

### Core Principle

The best hypothesis is NOT the one with the most supporting evidence, but the
one with the LEAST disconfirming evidence. Confirming evidence is weak —
most evidence is consistent with multiple hypotheses. Disconfirming evidence
is strong — a single well-established fact that contradicts a hypothesis
significantly weakens it.

### Procedure

1. **Enumerate candidate hypotheses**: List 4-6 mutually exclusive candidate
   explanations for the case. They must be mutually exclusive — if one
   is true, the others are false. Include:
   - The most obvious/intuitive explanation
   - At least one non-obvious alternative (organizational, design-related)
   - At least one "uncomfortable" hypothesis that stakeholders might resist
   - Consider including a deception/misreporting hypothesis if relevant

   For each hypothesis, write a one-sentence "plot" explaining the causal
   chain if this hypothesis is true.

2. **Compile diagnostic evidence**: Extract 6-10 specific evidence items from
   the context. Prioritize DIAGNOSTIC evidence — evidence that discriminates
   between hypotheses, not evidence that is consistent with all of them.
   Include:
   - Directly observed facts (timestamps, measurements, states)
   - Actions taken or not taken
   - Conditions and environmental factors
   - Notably ABSENT evidence (the "dog that didn't bark" — something that
     should be present if a hypothesis is true but isn't mentioned)
   - Key assumptions you are making

3. **Build the evidence-diagnosticity matrix**: Construct an explicit matrix
   with hypotheses as columns and evidence items as rows. For each cell,
   rate the relationship:

   | Rating | Meaning | Description |
   |--------|---------|-------------|
   | **C**  | Consistent | Evidence is expected if this hypothesis is true |
   | **Cs** | Strongly Consistent | Evidence is particularly compelling support |
   | **I**  | Inconsistent | Evidence is unlikely if this hypothesis is true |
   | **Is** | Strongly Inconsistent | Evidence would significantly disconfirm |
   | **NA** | Not Applicable | Evidence has no bearing on this hypothesis |

   Present the matrix in table format:

   ```
   | Evidence Item        | H1: ... | H2: ... | H3: ... | H4: ... |
   |----------------------|---------|---------|---------|---------|
   | E1: [description]    |   C     |   I     |   C     |   NA    |
   | E2: [description]    |   Cs    |   C     |   Is    |   C     |
   | ...                  |   ...   |   ...   |   ...   |   ...   |
   ```

   When a rating depends on an assumption, note the assumption explicitly.

4. **Calculate inconsistency scores**: For each hypothesis, count the
   weighted inconsistencies:
   - Each **I** = 1 point
   - Each **Is** = 2 points
   - **Inconsistency Score** = sum of I + 2×Is

   Rank hypotheses from lowest to highest inconsistency score. The
   hypothesis with the LOWEST score is tentatively most likely.

5. **Focus on diagnostic evidence**: Identify the most diagnostic evidence
   items — those where ratings vary most across hypotheses. Move these to
   the top of your analysis. Evidence that is C or NA for ALL hypotheses
   has zero diagnostic value and can be deprioritized.

6. **Sensitivity analysis**: For the top-ranked hypothesis, ask:
   - Which 1-2 evidence items are most critical ("load-bearing") for
     its ranking? If one of these items were wrong, misinterpreted,
     or removed, would the ranking change?
   - Are any load-bearing evidence items based on assumptions rather
     than directly observed facts?
   - If yes, the conclusion is FRAGILE — flag this explicitly.

7. **Identify future indicators**: Generate two lists:
   - **Confirming indicators**: 2-3 future observable events or evidence
     that would further confirm the top-ranked hypothesis.
   - **Disconfirming indicators**: 2-3 future observable events that
     would alert analysts that the top-ranked hypothesis is wrong and
     a different hypothesis should be reconsidered.

### Output Mapping

Translate the methodology's analytical output into the Hypothesis–Evidence
pairs your domain instructions specify; cardinality, category labels
(where applicable), and Evidence-anchoring rules are defined there. Use
the methodology's outputs as analytical inputs, not as output structure
— do not surface internal scaffolding (matrices, scoring tables,
step-by-step procedure, dimension lists) in the final output.

### Quality Criteria

- The evidence-diagnosticity matrix must be explicitly presented in table
  format, not described abstractly.
- Hypotheses must be genuinely mutually exclusive — overlapping hypotheses
  invalidate the matrix logic.
- Inconsistency scores must be calculated and reported for all hypotheses.
- Sensitivity analysis must identify specific load-bearing evidence and
  assess the fragility of the conclusion.
- Verification must cite the specific diagnostic evidence (where ratings
  differ across hypotheses), not generic supporting evidence.
- Acknowledge residual uncertainty where evidence is genuinely ambiguous.
