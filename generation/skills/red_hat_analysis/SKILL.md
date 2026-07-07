---
name: red_hat_analysis
description: >
  Red Hat Analysis. Generates hypotheses by adopting the perspective of the
  key actors involved — thinking as they would think, with their values,
  constraints, pressures, and information. Combats "mirror imaging" (assuming
  others think like us). Best suited for cases involving human decisions,
  organizational behavior, or adversarial actions where understanding the
  actors' perspective is critical to explaining the outcome.
---

## Analytical Framework: Red Hat Analysis

Apply Red Hat Analysis to generate hypotheses from the source context.
This method combats mirror imaging — the tendency to assume that actors
involved shared the analyst's values, knowledge, and decision-making
framework.

### Core Concept

Outcomes involving human decisions cannot be fully understood from an
external analyst's perspective alone. Red Hat Analysis requires stepping
into the shoes of the key actors — understanding their information state,
their pressures, their training, their organizational incentives — and
reasoning about how the situation looks from THEIR perspective, not yours.

### Procedure

1. **Identify key actors**: From the context, identify the 2-4 most critical
   decision-makers or actors whose actions influenced the outcome. These may
   include:
   - Frontline operators, researchers, or practitioners who took direct actions
   - Supervisors, managers, or leaders who set conditions or made oversight decisions
   - Designers, engineers, or architects who created the system or process
   - Organizational leaders who shaped culture, priorities, or resource allocation

2. **Reconstruct each actor's information state**: For each key actor, ask:
   - What did they KNOW at the time of their critical decision? (Not what
     we know now in hindsight.)
   - What did they NOT know? What information was unavailable to them?
   - What were their immediate pressures? (Time pressure, production
     pressure, social pressure, regulatory pressure)
   - What was their training and experience? How would that shape their
     mental model of the situation?

3. **Reason from their perspective**: For each actor:
   - Given their information state and pressures, was their action
     RATIONAL from their point of view, even if it looks like an error
     from ours?
   - What would YOU have done in their exact situation with their exact
     information? Be honest.
   - Where does their likely reasoning diverge from the "correct" action?
     Is the divergence due to information gaps, cognitive biases, training
     gaps, or systemic pressures?

4. **Identify perspective-driven hypotheses**: Based on the actor-perspective
   analysis, generate hypotheses that explain the outcome through the
   actors' decision logic rather than through external attribution of
   "error" or "negligence."

5. **Synthesize across perspectives**: Where do different actors' perspectives
   converge or conflict? Conflicts between actors' perspectives often
   reveal coordination failures, communication gaps, or systemic issues.

### Output Mapping

Translate the methodology's analytical output into the Hypothesis–Evidence
pairs your domain instructions specify; cardinality, category labels
(where applicable), and Evidence-anchoring rules are defined there. Use
the methodology's outputs as analytical inputs, not as output structure
— do not surface internal scaffolding (matrices, scoring tables,
step-by-step procedure, dimension lists) in the final output.

### Quality Criteria

- Each key actor's information state must be explicitly reconstructed,
  not assumed to be the same as the analyst's.
- "Human error" attributions must be re-examined from the actor's
  perspective — if the action was rational given their information,
  the root cause is the information gap, not the actor.
- Verification must distinguish between what the actor knew and what
  the analyst knows in hindsight.
- At least two distinct actor perspectives must be examined.
