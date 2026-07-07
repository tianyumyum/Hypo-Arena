---
name: brainstorming
description: >
  Structured Brainstorming. Generates hypotheses through a disciplined
  divergent-then-convergent process. Simulates multiple analytical perspectives
  to produce a broad set of candidate explanations before narrowing to the
  strongest. Best suited for cases where the evidence supports many possible
  interpretations and creative breadth is needed before analytical depth.
---

## Analytical Framework: Structured Brainstorming

Apply a systematic divergent-then-convergent brainstorming process to generate
hypotheses from the case context. This method combats premature closure
and anchoring by forcing broad idea generation before evaluation.

### Core Concept

Standard analysis tends to lock onto the first plausible explanation. Structured
brainstorming deliberately separates idea *generation* (divergent phase) from
idea *evaluation* (convergent phase). During generation, no idea is rejected;
during evaluation, ideas are grouped, compared, and ranked.

To simulate multi-perspective brainstorming with an LLM, adopt distinct
analytical viewpoints in sequence (e.g., domain specialist, systemic thinker,
organizational analyst, contrarian).

### Procedure

1. **Divergent phase — generate candidate explanations from multiple angles**:
   Adopt at least four distinct analytical perspectives and generate candidate
   hypotheses from each:
   - **Technical/mechanical perspective**: What physical, chemical, or
     engineering failure could explain the evidence?
   - **Human factors perspective**: What human errors, decisions, or cognitive
     biases could have contributed?
   - **Systemic/organizational perspective**: What management, procedural,
     or cultural failures could be underlying causes?
   - **Contrarian perspective**: What explanation would someone who disagrees
     with the obvious conclusion propose?

   For each perspective, generate 2-3 candidate hypotheses without filtering.
   Do not evaluate plausibility yet.

2. **Catalog all candidates**: List all generated hypotheses (expect 8-12 total)
   without ranking. Note which perspective generated each one.

3. **Convergent phase — group and evaluate**:
   - Group similar hypotheses into thematic clusters.
   - Identify "outlier" hypotheses that don't fit any cluster — these may be
     noise or may be seeds of genuinely novel insight. Examine them carefully.
   - For each cluster, evaluate: How well does this cluster explain the
     specific evidence in the context? What evidence supports it? What
     evidence is inconsistent with it?

4. **Select and refine**: From the clusters, select the hypotheses that:
   - Best explain the most diagnostic evidence
   - Are genuinely distinct from each other (not minor variations)
   - Cover different failure categories (mechanism, latent factor, intervention)

5. **Synthesize into output format**: Distill the selected hypotheses into
   the required structured output.

### Output Mapping

Translate the methodology's analytical output into the Hypothesis–Evidence
pairs your domain instructions specify; cardinality, category labels
(where applicable), and Evidence-anchoring rules are defined there. Use
the methodology's outputs as analytical inputs, not as output structure
— do not surface internal scaffolding (matrices, scoring tables,
step-by-step procedure, dimension lists) in the final output.

### Quality Criteria

- The divergent phase must produce genuinely distinct perspectives, not
  four versions of the same idea.
- The convergent phase must show explicit evaluation against evidence,
  not just subjective preference.
- Outlier hypotheses must be explicitly addressed — either incorporated
  or dismissed with reasoning.
- Verification must reference which analytical perspectives support the
  selected hypothesis and why alternative clusters were deprioritized.
