"""HypoArena rubric: dimensions, formulas, per-domain emphasis (shared by arena and score)."""

from __future__ import annotations

from basics import DomainConfig

PAIR_DIMENSIONS: dict[str, str] = {
    "grounding": (
        "Contextual Grounding — does the hypothesis follow logically from facts in "
        "the Context? Grounding is about reasoning warrant, not how many specific "
        "Context terms are echoed. An abstract hypothesis with tight causal "
        "reasoning can be better grounded than a verbose one that merely restates "
        "or decorates the Context. Claims that introduce specific mechanisms, "
        "entities, pathways, or details NOT present in the Context are speculative, "
        "not grounded — penalize them even when they sound plausible."
    ),
    "insight": (
        "Inferential Insight — does the hypothesis synthesize dispersed Context "
        "observations into a non-obvious but Context-constrained explanatory, "
        "predictive, or mechanistic claim? Insight is measured by depth of "
        "integration of what the Context actually provides, NOT by how far the "
        "claim reaches beyond the Context. Identifying a genuine tension or "
        "connection inside the Context is more insightful than inventing "
        "impressive-sounding details the Context cannot substantiate. Penalize "
        "speculative elaboration disguised as deeper analysis."
    ),
    "justification": (
        "Evidential Justification — does the Evidence concretely and proportionately "
        "support the Hypothesis? Claim strength must match support strength: a "
        "modest claim with strong support beats a bold claim with weak or "
        "speculative support. Penalize evidence chains that lean on assumptions "
        "the Context does not warrant, and penalize overclaiming where the "
        "conclusion exceeds what the evidence can bear."
    ),
}

SET_DIMENSIONS: dict[str, str] = {
    "breadth": (
        "Hypothesis-Space Breadth — does the set cover genuinely distinct, "
        "Context-supported analytical directions (different mechanisms, risk axes, "
        "causal pathways)? Breadth comes from the diversity of well-developed "
        "analytical angles, NOT from the total number of hypotheses or from "
        "extending into generic recommendations. A smaller set of well-developed "
        "directions is broader in meaningful terms than a larger set padded with "
        "shallow or formulaic entries."
    ),
    "distinctness": (
        "Directional Distinctness — are the hypotheses genuinely separable in their "
        "analytical angle, not redundant or trivially restated? Hypotheses that "
        "explore different facets, mechanisms, or implications of related phenomena "
        "are distinct as long as each offers a separately testable or investigable "
        "claim. Penalize inflating distinctness by mixing analytical categories "
        "when the extra categories lack Context-grounded substance."
    ),
    "utility": (
        "Analytical Utility — does the set help an analyst prioritize what to "
        "examine, test, or investigate next, based on the strength of the "
        "evidence-to-claim links? A set that surfaces fewer but more tightly "
        "warranted lines of inquiry is more useful than one that lists many "
        "loosely supported possibilities. Utility comes from investigative "
        "prioritization quality, not from sheer count."
    ),
}

PAIR_KEYS: tuple[str, ...] = tuple(PAIR_DIMENSIONS)
SET_KEYS: tuple[str, ...] = tuple(SET_DIMENSIONS)


# ---- reference-anchored diagnostic ----

RECALL_DIMENSION = (
    "Reference Recall — for each Hypothesis in the reference set, decide whether "
    "the Submission contains at least one Hypothesis that substantively covers "
    "the same core causal mechanism, claim, or conclusion. Mere topic overlap is "
    "not enough: the mechanism, claim, or conclusion must align. When `[category: "
    "...]` lane labels are present on both sides, treat shared lanes as a strong "
    "alignment signal but verify the underlying claim still matches. Report as "
    "hits/total where total is the count of reference Hypotheses and hits is the "
    "number the Submission covers. Recall is a diagnostic signal only; it does "
    "not feed the 1–5 dimension scores."
)


# ---- per-domain emphasis (shared by arena and score) ----

DOMAIN_EMPHASIS: dict[str, str] = {
    "biomedical_science": (
        "Prioritize mechanistic specificity and falsifiability. A hypothesis that "
        "names a specific pathway, cell type, or molecular mechanism is only "
        "stronger when those details are warranted by the Context — speculative "
        "mechanistic detail beyond the Context is a penalty, not a strength."
    ),
    "machine_learning": (
        "Prioritize methodological insight and experimental testability. A claim "
        "about a specific architectural change or training technique is stronger "
        "only when the Context provides evidence for why that approach would work; "
        "impressive-sounding technical proposals without Context warrant are "
        "speculative, not insightful."
    ),
    "social_science": (
        "Prioritize theoretical depth and empirical grounding. Naming a specific "
        "psychological mechanism or mediating variable is stronger only when the "
        "Context supports that specific mechanism; citing plausible-sounding "
        "constructs not anchored in the Context is speculation, not depth."
    ),
    "financial_analysis": (
        "Prioritize analytical insight grounded in disclosed figures and "
        "operational detail. Hypotheses must trace back to specific numbers, "
        "trends, or disclosures in the Context. Generic industry commentary, "
        "policy recommendations, or market speculation not anchored in the "
        "filing data should be penalized."
    ),
    "it_operations": (
        "Prioritize identification of specific failure mechanisms, architectural "
        "vulnerabilities, and diagnostic signals. Inferring likely causal "
        "mechanisms (rollouts, race conditions, cascading failures, resource "
        "exhaustion) from observed symptoms, timelines, and impact patterns is "
        "valid domain reasoning when the inferred mechanism logically explains the "
        "evidence. Penalize generic best-practice recommendations not tied to the "
        "specific incident."
    ),
    "safety_investigation": (
        "Prioritize identification of specific failure mechanisms and causal "
        "pathways. Inferring likely causal chains and latent conditions from the "
        "factual record (timelines, physical evidence, procedural context, and — "
        "distinctively — telling absences such as missing safeguards or "
        "unrecorded checks) is valid investigative reasoning when the inferred "
        "mechanism logically explains the observed outcomes. Penalize generic "
        "safety recommendations or industry-standard practices not anchored in "
        "the specific incident evidence."
    ),
}


def domain_emphasis(domain: DomainConfig) -> str:
    """Look up the per-domain emphasis text for a judge prompt."""
    if domain.name not in DOMAIN_EMPHASIS:
        raise KeyError(f"No emphasis for domain {domain.name!r}")
    return DOMAIN_EMPHASIS[domain.name]


# ---- scoring formulas ----

def compute_pair_score(scores: dict[str, float]) -> float:
    """q_i = mean over present pair-level dimensions for a single Hypothesis."""
    values = [float(scores[k]) for k in PAIR_KEYS if k in scores]
    return sum(values) / len(values) if values else 0.0


def compute_q_pair(per_pair: list[dict[str, float]]) -> float:
    """Q_pair = mean of q_i across K submitted pairs (paper §3.1.2)."""
    q_is = [compute_pair_score(p) for p in per_pair if p]
    return sum(q_is) / len(q_is) if q_is else 0.0


def compute_set_score(scores: dict[str, float]) -> float | None:
    """Q_set = mean over present set-level dimensions; None when no set scores were given."""
    values = [float(scores[k]) for k in SET_KEYS if k in scores]
    return sum(values) / len(values) if values else None


def compute_summary_score(qi: float, qset: float | None, *, multi: bool) -> float:
    """S = q_i for singletons; (Q_pair + Q_set) / 2 for multi-pair sets."""
    if not multi or qset is None:
        return qi
    return (qi + qset) / 2.0


def render_dimensions_block(domain: DomainConfig) -> str:
    """Render the rubric body for the judge: pair-level + (when applicable) set-level."""
    parts = ["Pair-level dimensions:"]
    for key, body in PAIR_DIMENSIONS.items():
        parts.append(f"  - {key}: {body}")
    if domain.multi_hypothesis:
        parts.append("Set-level dimensions (set as a whole):")
        for key, body in SET_DIMENSIONS.items():
            parts.append(f"  - {key}: {body}")
    return "\n".join(parts)


def render_recall_block() -> str:
    """Render the recall diagnostic block (only added when a reference set is supplied)."""
    return f"Reference-anchored diagnostic:\n  - recall: {RECALL_DIMENSION}"
