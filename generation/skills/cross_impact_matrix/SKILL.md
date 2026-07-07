---
name: cross_impact_matrix
description: >
  Cross-Impact Matrix Analysis. Systematically examines how each factor in
  the evidence interacts with every other factor, revealing reinforcing loops,
  inhibiting relationships, and overlooked connections. Best suited for cases
  with many interacting variables where the causal structure is unclear and
  hidden interactions may be driving the outcome.
---

## Analytical Framework: Cross-Impact Matrix Analysis

Apply cross-impact matrix analysis to generate hypotheses from the source
context. This method reveals hidden interactions between factors that simple
linear analysis misses.

### Core Concept

Complex outcomes arise not from a single cause but from the interaction of
multiple factors. A cross-impact matrix forces systematic examination of how
each factor affects every other factor, exposing reinforcing loops (where
factors amplify each other) and inhibiting relationships (where one factor
should have prevented or mitigated another but didn't).

### Procedure

1. **Extract key factors**: From the context, identify 5-8 key variables,
   conditions, or observations that may have contributed to the outcome or
   pattern. Include:
   - Technical, structural, or quantitative factors
   - Human actions, decisions, or behavioral patterns
   - Organizational, procedural, or systemic factors
   - Environmental, contextual, or temporal conditions

2. **Build the interaction matrix**: For each pair of factors (A, B), assess:
   - Does A's presence **enhance** or **amplify** B's impact?
   - Does A's presence **inhibit** or **reduce** B's impact?
   - Is the relationship **neutral** (no meaningful interaction)?
   - Note: the relationship may be **asymmetric** — A may enhance B while
     B inhibits A. Assess both directions.

   Use a simple notation:
   - **++** : strongly enhances
   - **+** : moderately enhances
   - **0** : neutral / no interaction
   - **-** : moderately inhibits
   - **--** : strongly inhibits

3. **Identify critical interactions**:
   - **Reinforcing loops**: Chains where A enhances B, B enhances C, and C
     enhances A — these can cause rapid escalation or self-sustaining patterns.
   - **Failed inhibitors**: Factor X should have counteracted Factor Y (a
     safeguard, control, or constraint) but didn't — why not?
   - **Overlooked connections**: Interactions that were not obvious but emerged
     from systematic pairwise examination.

4. **Construct causal narratives**: Using the critical interactions identified,
   build 2-3 explanatory narratives grounded in factor interactions rather
   than single-cause attribution.

5. **Identify leverage points**: From the matrix, determine which factor, if
   changed, would have the greatest cascading effect across the most
   interactions — this points toward the most impactful explanatory claim.

### Output Mapping

Translate the methodology's analytical output into the Hypothesis–Evidence
pairs your domain instructions specify; cardinality, category labels
(where applicable), and Evidence-anchoring rules are defined there. Use
the methodology's outputs as analytical inputs, not as output structure
— do not surface internal scaffolding (matrices, scoring tables,
step-by-step procedure, dimension lists) in the final output.

### Quality Criteria

- The matrix must include at least 5 factors and assess their pairwise
  interactions explicitly.
- Causal narratives must be built from interaction chains, not single factors.
- Failed inhibitors must be explicitly identified and explained.
- Verification must reference specific factor pairs and their interaction
  direction, not abstract reasoning.
