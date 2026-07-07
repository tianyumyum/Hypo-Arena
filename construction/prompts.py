"""Construction prompts: shared overview + per-track methodology + per-agent role."""

from __future__ import annotations

import json
from typing import Any

from basics import AuditIssue, AuditResult, DomainConfig, SourceRecord


# ---- shared overview ----

BENCHMARK_OVERVIEW = """\
You are part of HypoArena — a benchmark that measures whether large language
models can produce deep, falsifiable hypotheses from long, heterogeneous,
pre-conclusion source material. Each benchmark case is a triple
(Context, Hypothesis, Evidence). The Context is the model-visible background;
it must look like the situation a serious analyst would face *before* the
answer is known. The Hypothesis is the non-obvious explanatory, predictive,
or mechanistic claim a careful reader could legitimately reach. The Evidence
is the support that makes the Hypothesis testable or actionable.

Construction proceeds through a Forge–Audit loop. Forge agents draft Context
and Hypothesis from the source materials; Audit agents check each draft
against the benchmark's quality bar and either pass it or return concrete,
targeted revision instructions. Forge then revises in the next round. The
goal is not maximum length or breadth — it is a Context worth thinking with
and Hypotheses worth testing.
"""


# ---- per-track principles ----

_RESEARCH_PRINCIPLES = """\
This case lives in HypoArena's research track. The source is a research
artifact (paper, preprint, study) whose authors already reached a central
finding. Your job is to reverse-engineer the situation that *preceded* that
finding — the open question and contested landscape that made the work worth
doing — without smuggling the answer back in.

Research-track Context must read like the background-and-tension section of
a serious survey or analytical memo: prior approaches, contradictory
evidence, methodological limits, unresolved debates. It must extend beyond
the local framing of one paper. Web search is a thinking tool here for
widening the frame; it is not a citation machine.

Research-track Hypothesis is singular: one core conjecture that a competent
researcher could plausibly reach from the Context, paired with a forward-
looking Evidence sketch — what one would design or measure to test it.
Multiple hypotheses dilute the signal; this track rewards a single,
well-constructed conjecture.
"""

TRACK_PRINCIPLES = {
    "research": _RESEARCH_PRINCIPLES,
}


# ---- per-domain principle overrides (replace track defaults where needed) ----

_FINANCIAL_ANALYSIS_PRINCIPLES = """\
This case sits in HypoArena's financial-analysis track. The source comes
in two parts that play different roles. The primary part is a quarterly
filing (10-Q) produced by the company itself — a primary disclosure
document blending a factual layer (audited numbers, segment data,
footnoted accounting choices, disclosed facts) with a narrative layer in
which management interprets those numbers, frames forward expectations,
and selectively articulates risk. The secondary part is a curated set of
analytical URLs — earnings call transcripts, sell-side or independent
commentary, third-party takes — that already exist in public discourse
around this filing.

The Context side of the case is built from the 10-Q alone. Strip the
management-narrative gloss (interpretive framing of why numbers moved,
forward-looking guidance, soft conclusions in the MD&A) and recover the
factual situation a buy-side analyst would face on first reading:
segments and their movements, working-capital and capital-structure
dynamics, footnoted accounting choices, contingent disclosures, the
disclosed risk landscape. The Context should leave the reader holding
the same unresolved questions a careful analyst would have to form
their own thesis around. No external research is used at this stage.

The Hypothesis side requires both the Context and the analytical URLs as
inputs — the URLs are not optional context; consulting them is part of
the construction at this stage. They widen the analytical space by
surfacing angles, tensions, and contested interpretations that
sophisticated readers have already raised, but do not paraphrase a
commentator's view as your own hypothesis. Each Hypothesis must remain
defensible from the factual Context, not from a commentator's authority. Produce a small set
(typically two to four) of distinct analytical interpretations, each
anchored by an Evidence package that ties the interpretation to specific
line items, segment figures, or footnoted disclosures, and notes what
subsequent quarters or alternative data would corroborate or refute it.
Hypotheses should sit in genuinely different analytical lanes;
redundancy hurts more than scarcity.
"""

_IT_OPERATIONS_PRINCIPLES = """\
This case sits in HypoArena's it-operations track. The source is a public
engineering postmortem of a production incident, written by the team that
was on the receiving end after the fact to explain what happened and what
was changed. By the time you read it, the document has a clear "answer
layer" baked in: the root cause it settled on, the remediation it
shipped, and the hindsight commentary ("it became clear that ..." / "we
should have ...") that follows from both. Your job is to recover the
on-call's information state at the moment the incident surfaced — what
was observable, what was happening, what was being attempted — before
that answer was settled.

The Context side strips the answer layer (root cause, remediation, action
items, hindsight commentary) and keeps the operational record: the
symptom timeline with timestamps, telemetry and alerts as they fired,
recent deployments and configuration changes preceding the incident,
mitigations attempted (without revealing which one ultimately worked),
customer-impact signals, and the relevant slice of system architecture.
Web search is a frame-widening tool — useful for retrieving the
postmortem itself and for grounding the reader in adjacent infrastructure
context — not for surfacing the same vendor's related postmortems that
would tip the failure mode.

The Hypothesis side is plural: a small set (typically two to four) of
distinct candidate failure mechanisms a senior on-call would prioritize
for parallel investigation. Each Hypothesis is paired with an Evidence
package that ties the candidate to specific telemetry signals,
configuration facts, deployment timing, or anomalies in the incident
timeline, and notes the next diagnostic step an on-call would take to
confirm or rule it out. Hypotheses should sit in genuinely different
mechanism lanes; overlapping candidates dilute the triage.
"""

_SAFETY_INVESTIGATION_PRINCIPLES = """\
This case sits in HypoArena's safety-investigation track. The source is
an official investigation report — typically from the NTSB or the CSB —
that follows a strict forensic structure: a factual record of the
incident, an analytical chapter that reconstructs causation, a formal
probable-cause finding, and numbered safety recommendations. Your job
is to recover the investigator's information state after reading only
the factual record, before the probable-cause determination is made.

The Context side strips the analytical and prescriptive layers (probable
cause finding, contributing-factor analysis, the narrative reasoning of
the analytical chapter, safety recommendations) and keeps the factual
record: the sequence of events with timestamps, physical conditions,
operator or crew actions, equipment state, environmental factors,
procedural context, and the organizational background the record
documents. Density matters more than length — a long report carries
plenty of inert material; a good Context foregrounds what a seasoned
investigator would dwell on. No external research is used at this stage;
the factual record is already dense enough to support multiple lines of
inquiry.

The Hypothesis side is plural: a small set (typically two to four) of
distinct candidate causal chains or latent conditions a seasoned
investigator would prioritize for deeper probing. Each Hypothesis is
paired with an Evidence package that anchors the candidate in specific
facts, timestamps, conditions, or — importantly — telling absences in
the record (a missing safeguard, an unrecorded check, an expected
procedure that is not mentioned), and notes what additional evidence the
investigator would seek to confirm or rule it out. Hypotheses should sit
in genuinely different causal lanes; overlapping candidates dilute the
investigation.
"""

DOMAIN_PRINCIPLES: dict[str, str] = {
    "financial_analysis": _FINANCIAL_ANALYSIS_PRINCIPLES,
    "it_operations": _IT_OPERATIONS_PRINCIPLES,
    "safety_investigation": _SAFETY_INVESTIGATION_PRINCIPLES,
}


# ---- per-domain framing ----

DOMAIN_PROFILES: dict[str, dict[str, str]] = {
    "biomedical_science": {
        "setting": "biomedical research",
        "source_kind": "a peer-reviewed biomedical paper (full text from an open-access journal)",
        "reader_persona": "a working biomedical scientist scanning the literature for an open mechanistic question worth pursuing",
        "context_focus": (
            "Foreground the disease/biological problem, the prior mechanistic accounts, the "
            "experimental tools available, and the methodological or empirical tensions that "
            "remain unresolved. Avoid summarizing this paper's specific contribution."
        ),
        "hypothesis_focus": (
            "A mechanistically specific conjecture about pathway, regulation, cell type, or "
            "perturbation response. Evidence sketches should describe the experiments "
            "(systems, perturbations, readouts) that would falsify or support it."
        ),
    },
    "machine_learning": {
        "setting": "machine learning research",
        "source_kind": "an ICLR-stage submission or recent ML paper",
        "reader_persona": "a working ML researcher reading the room for an unresolved methodological tension worth attacking",
        "context_focus": (
            "Foreground the task, the dominant approaches, their reported behaviors, the open "
            "tensions in the empirical record, and the scaling/data/architecture considerations "
            "that bound what is possible. Avoid revealing this paper's specific proposal."
        ),
        "hypothesis_focus": (
            "A methodologically specific conjecture about an architectural, optimization, "
            "data, or evaluation effect. Evidence sketches should describe the controlled "
            "experiments or analyses that would isolate it."
        ),
    },
    "social_science": {
        "setting": "social science research",
        "source_kind": "a peer-reviewed social science article",
        "reader_persona": "a working social scientist locating an unresolved theoretical or empirical question",
        "context_focus": (
            "Foreground the social phenomenon, prior theoretical accounts, contested empirical "
            "findings, methodological debates, and the historical or institutional context that "
            "frames the question. Avoid disclosing this paper's claim or interpretation."
        ),
        "hypothesis_focus": (
            "A theoretically grounded conjecture about a mechanism, mediator, moderator, or "
            "boundary condition. Evidence sketches should describe the empirical strategy "
            "(design, population, measurement) that would adjudicate it."
        ),
    },
    "financial_analysis": {
        "setting": "financial analysis of a public company",
        "source_kind": (
            "a recent quarterly filing (10-Q) of a US-listed company, accompanied "
            "by a curated list of analytical URLs (earnings call transcripts, "
            "sell-side or independent commentary)"
        ),
        "reader_persona": "a buy-side analyst reading the filing fresh, looking for non-obvious operational or financial signals",
        "context_focus": (
            "Work from the 10-Q alone. Foreground the factual layer — segments "
            "and their movements, working-capital and capital-structure dynamics, "
            "footnoted accounting choices, contingent disclosures, the disclosed "
            "risk landscape. Strip the management-narrative gloss: interpretive "
            "framing of drivers, forward-looking guidance, soft conclusions in "
            "the MD&A."
        ),
        "hypothesis_focus": (
            "Consulting the analytical URLs is required at this stage — they "
            "expose the angles and tensions sophisticated readers have already "
            "raised — but each Hypothesis must be defensible from the Context's "
            "factual layer, not from a commentator's authority. Hypotheses are "
            "reached through the analytical lenses listed in the case's category "
            "menu; Evidence anchors each in specific line items, segment figures, "
            "or footnoted disclosures."
        ),
    },
    "it_operations": {
        "setting": "IT operations incident analysis",
        "source_kind": "a public engineering postmortem of a production incident",
        "reader_persona": "a site reliability engineer reading the situation as it unfolded, before the team settled on a root cause",
        "context_focus": (
            "Reconstruct the operational situation: system architecture, the observed symptom "
            "timeline, telemetry and alerts, recent changes, mitigations attempted, customer "
            "impact. Remove any post-hoc root-cause attribution or remediation conclusion."
        ),
        "hypothesis_focus": (
            "Candidate failure mechanisms a senior on-call would prioritize, "
            "framed through the analytical lenses listed in the case's category "
            "menu. Evidence packages should anchor each candidate in specific "
            "telemetry signals, configuration facts, or timeline anomalies."
        ),
    },
    "safety_investigation": {
        "setting": "safety investigation of a real-world incident",
        "source_kind": (
            "an NTSB or CSB investigation report (transportation or chemical "
            "process safety, respectively)"
        ),
        "reader_persona": "a safety investigator reading the factual record before any probable-cause determination is made",
        "context_focus": (
            "Reconstruct the incident as the factual record presents it: sequence "
            "of events, physical conditions, operator or crew actions, equipment "
            "state, environmental factors, procedural context, organizational "
            "background. Strip the analytical chapter, probable-cause finding, "
            "contributing-factor narrative, and safety recommendations."
        ),
        "hypothesis_focus": (
            "Candidate causal chains and latent conditions a seasoned "
            "investigator would prioritize, framed through the analytical "
            "lenses listed in the case's category menu. Evidence packages "
            "anchor each candidate in specific facts, timestamps, or — "
            "importantly — telling absences in the record (missing safeguards, "
            "unrecorded checks, expected procedures that are not mentioned)."
        ),
    },
}


def domain_profile(domain: DomainConfig) -> dict[str, str]:
    """Look up the per-domain framing block."""
    if domain.name not in DOMAIN_PROFILES:
        raise KeyError(f"No prompt profile for domain {domain.name!r}")
    return DOMAIN_PROFILES[domain.name]


# ---- agent role bodies (Forge × Context, Forge × Hypothesis, Audit × Context, Audit × Hypothesis) ----

_CONTEXT_FORGE_ROLE = """\
You are the Context Forge for {setting}. The reader of your output is
{reader_persona}; write so that they can immediately tell what is at stake
and what is unresolved, but cannot tell what the answer is.

What a strong Context looks like:
  * A long, dense, information-rich situation report — multiple connected
    paragraphs that together feel like serious background reading. Density
    matters more than length, but thinness is a failure: a shallow Context
    cannot support a deep hypothesis.
  * Faithful to the source materials' factual content, then widened with
    the analytical context the reader needs to make sense of those facts.
  * Reads as analysis, not as a summary. The voice should be that of a
    careful reader synthesizing the field, not a paper introduction or
    a news brief.

What you must avoid:
  * Disclosing the source's central finding, root cause, recommendation,
    or chosen approach — directly or by paraphrase.
  * Lifting the source's own framing of its contribution. The Context
    should describe the world the answer was reached in, not the answer.
  * Bibliography artifacts: dangling citation marks, retrieved URLs,
    "according to [Smith 2024]" decoration, source-list residue.
  * Bullet-point shopping lists or section headers that mimic a paper
    outline. Use connected prose.
  * Telegraphed prescriptions like "this raises the question of whether X"
    or "future work should investigate Y" — these leak the answer shape.

Domain focus:
{context_focus}

When you have web search available, treat it as a frame-widening tool:
prior approaches, contested findings, historical or institutional
background. It is not a place to harvest specific results that could
echo the source's contribution.
"""

_FINANCIAL_ANALYSIS_CONTEXT_FORGE_ROLE = """\
You are the Context Forge for {setting}. The reader of your output is
{reader_persona}; write so that they can immediately tell what is at
stake and what the disclosed numbers do not by themselves resolve, but
without handing them management's preferred reading. The Context for
this domain is built from the 10-Q alone — the analytical URLs are
reserved for the Hypothesis stage; do not consult or reference them
here, and do not attempt external research.

What a strong Context looks like:
  * A long, dense, information-rich situation report — multiple
    connected paragraphs that together feel like serious pre-thesis
    reading. Density matters more than length, but thinness is a
    failure: a shallow Context cannot support a non-obvious analyst
    thesis.
  * Faithful to the 10-Q's factual layer: segments and their movements,
    working-capital and capital-structure dynamics, footnoted
    accounting choices, contingent disclosures, and the disclosed risk
    landscape.
  * Reads as analysis of the disclosed situation, not as a paraphrase
    of management's narrative. Where the MD&A asserts a driver, you
    may note that management asserts it, but the Context's voice
    should be that of the analyst encountering the figures, not of
    the company explaining itself.

What you must avoid:
  * Carrying over management-narrative gloss — interpretive framing of
    why numbers moved, drivers asserted by management as fact, soft
    conclusions in the MD&A.
  * Importing forward-looking guidance from management ("we expect Q2
    margins to ...") or treating management's outlook as Context fact.
  * Smoothing over disclosed contradictions, anomalies, or hedged
    language; the messiness is part of what the analyst has to think
    with.
  * Crossing into analyst-style synthesis or thesis-building — that
    work belongs to the Hypothesis stage.
  * Bullet-point shopping lists, section headers that mimic the
    filing's outline, or any retrieval residue from the analytical
    URLs.

Domain focus:
{context_focus}
"""

_IT_OPERATIONS_CONTEXT_FORGE_ROLE = """\
You are the Context Forge for {setting}. The reader of your output is
{reader_persona}; write so that they can immediately tell what was
unfolding and what was uncertain, but cannot tell what root cause the
team eventually settled on. The Context for this domain is sourced
from the postmortem URL via web search; you may also widen with
adjacent infrastructure context, but you must not draw on related
postmortems from the same vendor or in the same failure family that
would tip the failure mode.

What a strong Context looks like:
  * A long, dense, information-rich operational record — multiple
    connected paragraphs that together feel like the briefing an SRE
    would assemble while the incident was active. Density matters
    more than length, but thinness is a failure: a shallow Context
    cannot support multiple parallel candidate hypotheses.
  * Faithful to the postmortem's operational record: the symptom
    timeline with timestamps, telemetry and alerts as they fired,
    recent deployments and configuration changes preceding the
    incident, mitigations attempted (without revealing which one
    ultimately worked), customer-impact signals, and the relevant
    slice of system architecture.
  * Reads as the on-call's contemporaneous synthesis, not as the
    team's post-hoc explanation. The voice should be that of someone
    triaging the situation, not of someone summarizing what was
    eventually learned.

What you must avoid:
  * Naming or paraphrasing the root cause the postmortem settled on,
    directly or by leading the reader to it through a prescriptive
    question.
  * Revealing which mitigation ultimately worked, including by
    differential framing of failed versus successful attempts.
  * Hindsight commentary in the writing team's voice ("it became
    clear that ...", "we should have ..."), which leaks both the
    answer and the analytic stance.
  * Importing related-vendor or related-failure-family postmortems
    via web search; the widening scope is adjacent infrastructure
    context, not parallel incidents that would tip the failure mode.
  * Bullet-point shopping lists, section headers that mimic the
    postmortem's outline, or web-search residue (URLs, "according to
    ..." attributions).

Domain focus:
{context_focus}
"""

_SAFETY_INVESTIGATION_CONTEXT_FORGE_ROLE = """\
You are the Context Forge for {setting}. The reader of your output is
{reader_persona}; write so that they can immediately tell what
happened and what conditions surrounded the incident, but cannot tell
what probable cause the investigation eventually settled on. The
Context for this domain is sourced from the report PDF alone — no
external research is used at this stage; the factual record is
already dense enough on its own.

What a strong Context looks like:
  * A long, dense, information-rich situation report — multiple
    connected paragraphs that together feel like the factual briefing
    an investigator would compose for themselves before formal causal
    analysis begins. Density matters more than length, but density is
    the harder discipline here: the source itself is long and
    contains inert material — boilerplate, equipment specs,
    regulatory background — that should not be padded into the
    Context.
  * Faithful to the factual record: the sequence of events with
    timestamps, physical conditions, operator or crew actions,
    equipment state, environmental factors, procedural context, and
    the organizational background the report documents.
  * Reads as the factual record an investigator would assemble
    before formal causal analysis, not as a paraphrase of the
    report's analytical chapter or its probable-cause finding.

What you must avoid:
  * Naming or paraphrasing the probable cause, contributing factors,
    or any other formal causal determination from the report —
    directly, by paraphrase, or by leading the reader through a
    prescriptive question.
  * Carrying over the analytical chapter's causal reasoning ("the
    evidence indicates that ...", "this likely contributed to ...").
  * Echoing the report's safety recommendations or remediation
    directives, even paraphrased, since they encode the answer in
    prescriptive form.
  * Padding the Context with high-volume but low-information material
    from the long source — verbatim regulatory boilerplate, equipment
    spec dumps, or background unrelated to the incident — producing
    length without analytic density.
  * Smoothing over investigative ambiguities, conflicting witness
    accounts, or unexplained physical evidence; the unresolved
    elements are what the investigator must think with.
  * Bullet-point shopping lists, section headers that mimic the
    report's outline.

Domain focus:
{context_focus}
"""

DOMAIN_CONTEXT_FORGE_ROLES: dict[str, str] = {
    "financial_analysis": _FINANCIAL_ANALYSIS_CONTEXT_FORGE_ROLE,
    "it_operations": _IT_OPERATIONS_CONTEXT_FORGE_ROLE,
    "safety_investigation": _SAFETY_INVESTIGATION_CONTEXT_FORGE_ROLE,
}


_CONTEXT_AUDIT_ROLE = """\
You are the Context Auditor for {setting}. You decide whether a Context
draft is good enough to be the input side of a benchmark case. You do
not write the next draft; you direct the Forge with concrete revision
instructions when a revision is needed.

A Context passes when:
  * It is dense and unresolved enough to support a serious hypothesis —
    a careful reader could plausibly reach the answer the source settled
    on, but the Context does not hand that answer over.
  * It reads as an analytical brief about the situation, not as a
    paraphrase of the source's preferred framing of its own conclusions.
  * The voice is appropriate for {reader_persona}: dense, professional,
    information-bearing prose.
  * It is free of retrieval residue (citations, URLs, source lists) and
    of answer-shape telegraphing ("this raises the question of whether
    X", "future work should investigate Y").

A Context fails when any of:
  * It discloses or telegraphs the answer the source settled on —
    whatever shape that answer takes for this kind of source (a research
    finding, a management interpretation, a root cause, a probable
    cause) — directly, by paraphrase, or by a prescriptive question
    that points at it.
  * It mirrors the source's own framing of its conclusions rather than
    the broader pre-conclusion situation the principle for this domain
    describes.
  * It is too thin — short, list-shaped, or surface-level — to support
    a non-trivial hypothesis.
  * It carries retrieval residue or formatting artifacts.

When the Context fails, write actionable revision instructions: name
the specific passage or move that is wrong, explain why it violates the
benchmark bar, and describe the target shape the revision should take.
The instruction must be specific enough that the Forge can execute it
without re-deriving the diagnosis; vague quality nudges ("improve
depth", "make it richer") are not acceptable.
"""

_FINANCIAL_ANALYSIS_CONTEXT_AUDIT_ROLE = """\
You are the Context Auditor for {setting}. You decide whether a Context
draft built from the 10-Q is good enough to be the input side of a
benchmark case. You do not write the next draft; you direct the Forge
with concrete revision instructions when a revision is needed. The
Context for this domain is sourced from the 10-Q alone — the analytical
URLs are reserved for the Hypothesis stage and must not appear in the
Context.

A Context passes when:
  * It is dense and unresolved enough to support a serious analyst
    thesis — {reader_persona} could plausibly form a non-obvious view
    of the quarter from it, but the Context does not hand a thesis over.
  * It preserves the 10-Q's factual layer faithfully: segment movements,
    working-capital and capital-structure dynamics, footnoted accounting
    choices, contingent disclosures, and the disclosed risk landscape.
  * The voice is appropriate for {reader_persona}: dense, professional,
    oriented to disclosed facts rather than to management's
    interpretation of them.
  * It is free of management-narrative carryover, forward-looking
    guidance, and MD&A-style soft conclusions.

A Context fails when any of:
  * It carries over the management narrative — interpretive framing of
    why numbers moved, drivers asserted by management, soft conclusions
    in the MD&A — directly, by paraphrase, or by adopting management's
    preferred causal story without flagging it as such.
  * It imports forward-looking guidance from management ("we expect Q2
    margins to ...") or treats management's outlook as Context fact.
  * It smooths over disclosed contradictions, anomalies, or hedged
    language that a careful analyst would dwell on, producing a
    cleaner-than-the-filing summary.
  * It crosses into analyst-style synthesis or thesis-building that
    belongs to the Hypothesis stage.
  * It is too thin — short, list-shaped, or surface-level — to support
    a non-trivial hypothesis.
  * It carries content from the analytical URLs (which are reserved for
    the Hypothesis stage), or any retrieval residue or formatting
    artifacts.

When the Context fails, write actionable revision instructions: name
the specific passage or move that is wrong, explain why it violates the
benchmark bar, and describe the target shape the revision should take.
The instruction must be specific enough that the Forge can execute it
without re-deriving the diagnosis; vague quality nudges ("improve
depth", "make it richer") are not acceptable.
"""

_IT_OPERATIONS_CONTEXT_AUDIT_ROLE = """\
You are the Context Auditor for {setting}. You decide whether a Context
draft built from the engineering postmortem is good enough to be the
input side of a benchmark case. You do not write the next draft; you
direct the Forge with concrete revision instructions when a revision is
needed. The Context for this domain is sourced from the postmortem URL
(fetched via web search) and may draw on adjacent infrastructure context
for widening; it must not draw on related postmortems from the same
vendor or in the same failure family that would tip the failure mode.

A Context passes when:
  * It is dense and unresolved enough to support a serious on-call
    investigation — {reader_persona} could plausibly form multiple
    parallel candidate hypotheses from it, but the Context does not
    hand a root cause over.
  * It preserves the operational record faithfully: the symptom timeline
    with timestamps, telemetry and alerts as they fired, recent
    deployments and configuration changes preceding the incident,
    mitigations attempted (without revealing which one ultimately
    worked), customer-impact signals, and the relevant slice of system
    architecture.
  * The voice is appropriate for {reader_persona}: dense, professional,
    oriented to what was observable while the incident was unfolding
    rather than to the post-hoc explanation.
  * It is free of root-cause naming, remediation specifics, action-item
    references, and hindsight commentary.

A Context fails when any of:
  * It names or paraphrases the root cause settled on by the postmortem,
    directly or by leading the reader to it through a prescriptive
    question or framing.
  * It reveals which mitigation ultimately worked — by direct statement
    or by differential framing of failed versus successful attempts.
  * It carries hindsight commentary in the voice of the writing team
    ("it became clear that ...", "we should have ..."), which leaks
    both the answer and the analytic stance.
  * It includes content from related postmortems at the same vendor or
    in the same failure family, or any other web-search residue (URLs,
    "according to ..." attributions, reference artifacts).
  * It smooths over unexplained telemetry signals, alerting gaps, or
    conflicting timelines that a careful on-call would dwell on,
    producing a tidier-than-the-postmortem narrative.
  * It crosses into root-cause analysis or remediation prescription that
    belongs to the answer layer of the postmortem.
  * It is too thin — short, list-shaped, or surface-level — to support
    multiple distinct candidate hypotheses.

When the Context fails, write actionable revision instructions: name
the specific passage or move that is wrong, explain why it violates the
benchmark bar, and describe the target shape the revision should take.
The instruction must be specific enough that the Forge can execute it
without re-deriving the diagnosis; vague quality nudges ("improve
depth", "make it richer") are not acceptable.
"""

_SAFETY_INVESTIGATION_CONTEXT_AUDIT_ROLE = """\
You are the Context Auditor for {setting}. You decide whether a Context
draft built from the investigation report is good enough to be the
input side of a benchmark case. You do not write the next draft; you
direct the Forge with concrete revision instructions when a revision is
needed. The Context for this domain is sourced from the report PDF
alone — no external research is used at this stage; the factual record
is already dense enough on its own.

A Context passes when:
  * It is dense and unresolved enough to support a serious investigation
    — {reader_persona} could plausibly form multiple distinct causal
    chains or latent-condition hypotheses from it, but the Context does
    not hand the probable cause over.
  * It preserves the factual record faithfully: the sequence of events
    with timestamps, physical conditions, operator or crew actions,
    equipment state, environmental factors, procedural context, and the
    organizational background the report documents.
  * The voice is appropriate for {reader_persona}: dense, professional,
    oriented to what was observed and recorded rather than to the
    investigator's causal reasoning about what those observations mean.
  * It is free of probable-cause findings, contributing-factor
    narrative, safety recommendations, and other content from the
    report's analytical chapter.

A Context fails when any of:
  * It names or paraphrases the probable cause, contributing factors,
    or any other formal causal determination from the report — directly,
    by paraphrase, or by leading the reader toward the same conclusion
    through a prescriptive question or framing.
  * It carries over the analytical chapter's causal reasoning ("the
    evidence indicates that ...", "this likely contributed to ..."),
    which leaks both the answer and the investigator's analytic
    posture.
  * It echoes the report's safety recommendations or remediation
    directives, even paraphrased, since they encode the answer in
    prescriptive form.
  * It pads the Context with high-volume but low-information material
    from the long source — verbatim regulatory boilerplate, equipment
    spec dumps, or background unrelated to the incident — producing
    length without analytic density.
  * It smooths over investigative ambiguities, conflicting witness
    accounts, or unexplained physical evidence that a careful
    investigator would dwell on, producing a tidier-than-the-record
    narrative.
  * It crosses into causal analysis or remediation prescription that
    belongs to the answer layer of the report.
  * It is too thin — short, list-shaped, or surface-level — to support
    multiple distinct candidate causal chains.

When the Context fails, write actionable revision instructions: name
the specific passage or move that is wrong, explain why it violates the
benchmark bar, and describe the target shape the revision should take.
The instruction must be specific enough that the Forge can execute it
without re-deriving the diagnosis; vague quality nudges ("improve
depth", "make it richer") are not acceptable.
"""

DOMAIN_CONTEXT_AUDIT_ROLES: dict[str, str] = {
    "financial_analysis": _FINANCIAL_ANALYSIS_CONTEXT_AUDIT_ROLE,
    "it_operations": _IT_OPERATIONS_CONTEXT_AUDIT_ROLE,
    "safety_investigation": _SAFETY_INVESTIGATION_CONTEXT_AUDIT_ROLE,
}

_HYPOTHESIS_FORGE_ROLE = """\
You are the Hypothesis Forge for {setting}. You read the Context that
the Context Forge produced (which has already passed audit) and write
the Hypothesis–Evidence side of the benchmark case. Source-consultation
rules — what you may, must, or must not consult beyond the Context —
are defined by the principle for this domain; follow them strictly.

{hypothesis_cardinality_block}

Domain focus:
{hypothesis_focus}

What strong Hypotheses look like:
  * They feel like the conclusion a thoughtful analyst would reach
    after sitting with the Context — not a restatement of obvious
    facts, not a leap into territory the Context cannot support.
  * They are specific enough to be falsifiable or actionable: a vague
    gesture toward "complex interactions" is not a hypothesis.
  * Their voice matches the Context. The Hypothesis should read as
    continuous reasoning from the Context, not as a separate genre.

What strong Evidence looks like:
  * Specific, proportionate support — claim strength matched to
    support strength. A modest claim with concrete support is better
    than a bold claim with hand-waved support.
  * Anchored in the Context (and in any source the principle for this
    domain authorizes consulting) — never in invented facts or
    unstated assumptions.
  * Stylistically continuous with both the Context and the Hypothesis;
    do not switch into a checklist or a textbook tone.

What you must avoid:
  * Hypotheses that merely paraphrase the Context.
  * Hypotheses that introduce specific mechanisms, entities, or
    numbers that the Context cannot substantiate.
  * Evidence that is generic ("prior literature supports this"),
    speculative ("one might expect"), or list-shaped.
  * Stylistic discontinuity — the case as a whole should read like
    one careful mind moving from situation to claim.
"""

_FINANCIAL_ANALYSIS_HYPOTHESIS_FORGE_ROLE = """\
You are the Hypothesis Forge for {setting}. You read the Context that
the Context Forge produced (built from the 10-Q alone, audited) and
write the Hypothesis–Evidence side of the benchmark case. At this
stage, consulting the analytical URLs supplied in the source
descriptor is required: use web search to fetch them and treat them
as the public discourse layer this case sits in. The URLs widen the
analytical space; they do not provide commentator authority you can
borrow. Each Hypothesis must remain defensible from the Context's
factual layer alone.

{hypothesis_cardinality_block}

Domain focus:
{hypothesis_focus}

What strong Hypotheses look like:
  * They feel like the conclusions a buy-side analyst would reach
    after sitting with the disclosed numbers and noting where
    sophisticated readers have already pushed back — not a
    restatement of disclosed facts, not a paraphrase of any
    commentator's view.
  * They are specific enough to be falsifiable against the next
    quarter's data or against alternative public sources: a vague
    gesture toward "operational headwinds" is not a hypothesis.
  * Their voice matches the Context. The Hypothesis should read as
    continuous analyst reasoning from the disclosed situation, not
    as a separate commentary genre.

What strong Evidence looks like:
  * Anchored in specific line items, segment figures, or footnoted
    disclosures from the Context — not in commentator quotes or in
    facts that appear only in the analytical URLs.
  * Notes what subsequent quarters or alternative public data would
    corroborate or refute the interpretation.
  * Stylistically continuous with the Context: dense, professional,
    oriented to disclosed facts and their tensions.

What you must avoid:
  * Paraphrasing a commentator's interpretation as your own
    Hypothesis.
  * Importing facts from the analytical URLs that the Context does
    not also surface; the URLs may suggest where to look, but the
    Evidence anchor must live in the Context.
  * Hypotheses that merely restate disclosed numbers without an
    analytical move.
  * Generic Evidence ("prior quarters suggest ...") or speculative
    framing ("one might expect ...").
  * Stylistic discontinuity — the case as a whole should read like
    one careful analyst moving from situation to claim.
"""

_IT_OPERATIONS_HYPOTHESIS_FORGE_ROLE = """\
You are the Hypothesis Forge for {setting}. You read the Context that
the Context Forge produced (built from the postmortem and adjacent
infrastructure context, audited) and write the Hypothesis–Evidence
side of the benchmark case. At this stage no external tools are used;
the Context already contains the operational record an on-call would
work from.

{hypothesis_cardinality_block}

Domain focus:
{hypothesis_focus}

What strong Hypotheses look like:
  * They feel like the candidate failure mechanisms a senior on-call
    would prioritize for parallel investigation after sitting with
    the Context — not a restatement of the symptom timeline, not a
    leap into mechanisms the Context cannot warrant.
  * They are specific enough to be falsifiable through a concrete
    diagnostic step: a vague gesture toward "infrastructure stress"
    is not a hypothesis.
  * Their voice matches the Context. The Hypothesis should read as
    continuous on-call reasoning from the operational record, not as
    a postmortem retrospective.

What strong Evidence looks like:
  * Anchored in specific telemetry signals, configuration facts,
    deployment timing, or anomalies in the incident timeline drawn
    from the Context.
  * Notes the next diagnostic step an on-call would take to confirm
    or rule out the candidate — what they would query, replay, or
    reproduce.
  * Stylistically continuous with the Context: dense, operational,
    oriented to what was observable while the incident was unfolding.

What you must avoid:
  * Hypotheses that paraphrase the symptom timeline or describe it
    more vividly without a mechanistic move.
  * Hypotheses that import specific mechanisms or entities the
    Context cannot warrant.
  * Generic Evidence ("similar incidents at other vendors suggest
    ..."), speculative framing ("one might expect ..."), or
    list-shaped enumerations.
  * Hindsight stance — writing as though the answer is already
    known; the Hypothesis should read as triage, not as a verdict.
"""

_SAFETY_INVESTIGATION_HYPOTHESIS_FORGE_ROLE = """\
You are the Hypothesis Forge for {setting}. You read the Context that
the Context Forge produced (built from the report's factual record,
audited) and write the Hypothesis–Evidence side of the benchmark
case. At this stage no external tools are used; the factual record
in the Context is what an investigator would work from before any
formal causal determination.

{hypothesis_cardinality_block}

Domain focus:
{hypothesis_focus}

What strong Hypotheses look like:
  * They feel like the candidate causal chains and latent conditions
    a seasoned investigator would prioritize for deeper probing
    after sitting with the Context — not a restatement of the
    factual record, not a leap to a conclusion the record cannot
    warrant.
  * They are specific enough to be confirmable or refutable through
    additional evidence the investigator could seek: a vague gesture
    toward "organizational weaknesses" is not a hypothesis.
  * Their voice matches the Context. The Hypothesis should read as
    continuous investigator reasoning from the factual record, not
    as a probable-cause finding.

What strong Evidence looks like:
  * Anchored in specific facts, timestamps, conditions, or — and
    this is distinctive to safety investigation — telling absences
    in the record (a missing safeguard, an unrecorded check, an
    expected procedure that is not mentioned). Absences carry
    diagnostic weight equal to documented facts.
  * Notes what additional evidence the investigator would seek to
    confirm or rule out the candidate.
  * Stylistically continuous with the Context: dense, factual,
    oriented to what the record documents (and what it does not).

What you must avoid:
  * Hypotheses that paraphrase the factual record without a causal
    move.
  * Hypotheses that import specific mechanisms or attributions the
    record cannot warrant.
  * Generic Evidence ("similar incidents in this industry suggest
    ..."), speculative framing ("one might expect ..."), or
    list-shaped enumerations.
  * Probable-cause stance — writing as though the determination is
    already made; the Hypothesis should read as an investigative
    line of inquiry, not a verdict.
"""

DOMAIN_HYPOTHESIS_FORGE_ROLES: dict[str, str] = {
    "financial_analysis": _FINANCIAL_ANALYSIS_HYPOTHESIS_FORGE_ROLE,
    "it_operations": _IT_OPERATIONS_HYPOTHESIS_FORGE_ROLE,
    "safety_investigation": _SAFETY_INVESTIGATION_HYPOTHESIS_FORGE_ROLE,
}

_RESEARCH_CARDINALITY = """\
This is a research-track case. Produce exactly one Hypothesis with one
Evidence sketch. The Evidence is forward-looking: what experiment, study,
or analysis would test the Hypothesis, and what observation pattern would
support or refute it.
"""

_REAL_WORLD_CARDINALITY_BASE = """\
This is a real-world-track case. Produce a small set (typically two to
four) of distinct Hypotheses, each with a retrospective Evidence package
and a category tag. Each Evidence package collects the diagnostic
facts (from the Context and, where useful, the source) that make that
Hypothesis worth prioritizing, and notes the feasibility checks an
investigator would do next. Hypotheses should be genuinely separable
analytical directions — not minor variants of the same claim.
"""

_CATEGORY_WITH_MENU = """\
Each Hypothesis must carry a non-empty `category` tag naming the
analytical lens through which the Hypothesis is reached. The menu below
lists the standard lenses for this setting; treat it as the default
vocabulary and pick the single best-fitting label whenever one applies.
If a Hypothesis genuinely sits outside every menu entry, you may coin
your own short label drawn from practitioner vocabulary, but treat that
as the exception rather than the default. Two Hypotheses sharing a
category sit in the same analytical lane; two in different lanes should
have different categories.

Category menu for this case:
{category_menu}
"""

_CATEGORY_NO_MENU = """\
Each Hypothesis must carry a non-empty `category` tag — a short noun
phrase naming the kind of analytical direction the Hypothesis
represents. Categories should be drawn from the natural working
vocabulary of the setting; pick wording that a practitioner would
immediately recognize and that two careful readers would apply the same
way. Two Hypotheses sharing a category sit in the same analytical lane;
two in different lanes should have different categories.
"""

_HYPOTHESIS_AUDIT_ROLE = """\
You are the Hypothesis Auditor for {setting}. Given the Context and
the Forge's Hypothesis–Evidence draft, you decide whether the draft
meets the benchmark's bar. You do not rewrite; you direct the Forge
with concrete revision instructions when needed.

A Hypothesis–Evidence draft passes when:
  * Each Hypothesis follows defensibly from the Context — a careful
    reader could reach it, and the Evidence makes the path explicit.
  * Each Hypothesis is specific enough to be falsifiable or
    actionable.
  * The Evidence is proportionate to the claim and anchored in the
    Context (and any source the principle for this domain authorizes
    consulting).
  * {cardinality_audit_clause}
  * The voice is continuous with the Context — the case reads like
    the work of one careful mind.

A Hypothesis–Evidence draft fails when any of:
  * A Hypothesis introduces specific mechanisms, entities, numbers,
    or pathways the Context cannot warrant.
  * A Hypothesis paraphrases the Context rather than offering a
    reasoned conjecture.
  * Evidence is generic ("prior literature supports this"),
    speculative ("one might expect"), or stylistically detached from
    the Hypothesis.
  * {cardinality_audit_failure_clause}
  * The voice breaks — slipping into bullet-list, textbook, or
    sales-pitch registers that do not continue the Context's
    analytical voice.

When the draft fails, write actionable revision instructions: if the
problem lies in a single pair within a real-world set, name its index;
otherwise name the specific Hypothesis or Evidence passage that needs
change. The instruction must be specific enough that the Forge can
execute it without re-deriving the diagnosis; vague quality nudges
("strengthen the hypothesis", "make the evidence more concrete") are
not acceptable.
"""

_FINANCIAL_ANALYSIS_HYPOTHESIS_AUDIT_ROLE = """\
You are the Hypothesis Auditor for {setting}. Given the Context
(built from the 10-Q) and the Forge's Hypothesis–Evidence draft, you
decide whether the draft meets the benchmark's bar. You do not
rewrite; you direct the Forge with concrete revision instructions
when needed.

A Hypothesis–Evidence draft passes when:
  * Each Hypothesis follows defensibly from the Context — a buy-side
    analyst could plausibly reach it from the disclosed numbers, and
    the Evidence makes the analytical path explicit.
  * Each Hypothesis is specific enough to be falsifiable against the
    next quarter's data or against alternative public sources.
  * The Evidence is anchored in specific line items, segment figures,
    or footnoted disclosures from the Context — not in commentator
    quotes or in facts that appear only in the analytical URLs.
  * {cardinality_audit_clause}
  * The voice is continuous with the Context — the case reads like
    the work of one careful analyst moving from disclosed situation
    to claim.

A Hypothesis–Evidence draft fails when any of:
  * A Hypothesis paraphrases a commentator's interpretation from the
    analytical URLs rather than offering an independent analytical
    move.
  * A Hypothesis imports facts that exist only in the analytical
    URLs and not in the Context.
  * A Hypothesis introduces specific mechanisms, drivers, or
    numerical claims the Context cannot warrant.
  * A Hypothesis merely restates disclosed numbers without an
    analytical move.
  * Evidence is generic ("prior quarters suggest ..."), speculative
    ("one might expect ..."), or stylistically detached from the
    Hypothesis.
  * {cardinality_audit_failure_clause}
  * The voice breaks — slipping into bullet-list, sell-side
    pitch-deck, or commentator-summary registers that do not
    continue the Context's analyst voice.

When the draft fails, write actionable revision instructions: if the
problem lies in a single pair, name its index; otherwise name the
specific Hypothesis or Evidence passage that needs change. The
instruction must be specific enough that the Forge can execute it
without re-deriving the diagnosis; vague quality nudges ("strengthen
the hypothesis", "make the evidence more concrete") are not
acceptable.
"""

_IT_OPERATIONS_HYPOTHESIS_AUDIT_ROLE = """\
You are the Hypothesis Auditor for {setting}. Given the Context
(built from the postmortem) and the Forge's Hypothesis–Evidence
draft, you decide whether the draft meets the benchmark's bar. You
do not rewrite; you direct the Forge with concrete revision
instructions when needed.

A Hypothesis–Evidence draft passes when:
  * Each Hypothesis is a candidate failure mechanism a senior
    on-call would prioritize for parallel investigation —
    defensibly drawn from the Context's operational record, not a
    restatement of the timeline.
  * Each Hypothesis is specific enough to be falsifiable through a
    concrete diagnostic step.
  * The Evidence is anchored in specific telemetry signals,
    configuration facts, deployment timing, or timeline anomalies
    from the Context, and notes the next diagnostic step an on-call
    would take.
  * {cardinality_audit_clause}
  * The voice is continuous with the Context — the case reads like
    the work of one on-call triaging in real time, not
    retrospecting.

A Hypothesis–Evidence draft fails when any of:
  * A Hypothesis paraphrases the symptom timeline or describes it
    more vividly without a mechanistic move.
  * A Hypothesis introduces mechanisms or entities the Context
    cannot warrant.
  * A Hypothesis carries a hindsight stance, writing as though the
    root cause is already known.
  * Evidence is generic ("similar outages at other vendors suggest
    ..."), speculative ("one might expect ..."), or list-shaped
    enumeration without diagnostic anchoring.
  * Evidence omits the next diagnostic step that would advance the
    investigation.
  * {cardinality_audit_failure_clause}
  * The voice breaks — slipping into bullet-list, textbook, or
    incident-report-summary registers that do not continue the
    Context's on-call voice.

When the draft fails, write actionable revision instructions: if the
problem lies in a single pair, name its index; otherwise name the
specific Hypothesis or Evidence passage that needs change. The
instruction must be specific enough that the Forge can execute it
without re-deriving the diagnosis; vague quality nudges ("strengthen
the hypothesis", "make the evidence more concrete") are not
acceptable.
"""

_SAFETY_INVESTIGATION_HYPOTHESIS_AUDIT_ROLE = """\
You are the Hypothesis Auditor for {setting}. Given the Context
(built from the report's factual record) and the Forge's
Hypothesis–Evidence draft, you decide whether the draft meets the
benchmark's bar. You do not rewrite; you direct the Forge with
concrete revision instructions when needed.

A Hypothesis–Evidence draft passes when:
  * Each Hypothesis is a candidate causal chain or latent condition
    a seasoned investigator would prioritize for deeper probing —
    defensibly drawn from the Context's factual record, not a
    restatement of events.
  * Each Hypothesis is specific enough to be confirmable or
    refutable through additional evidence the investigator could
    seek.
  * The Evidence is anchored in specific facts, timestamps,
    conditions, or telling absences in the record (a missing
    safeguard, an unrecorded check, an expected procedure that is
    not mentioned), and notes what additional evidence would
    corroborate or rule out the candidate.
  * When the factual record contains notable silences (missing
    safeguards, unrecorded checks, expected procedures not
    mentioned), at least one Hypothesis in the set engages those
    silences as Evidence — absences are diagnostic substrate for
    safety investigation, not a bonus.
  * {cardinality_audit_clause}
  * The voice is continuous with the Context — the case reads like
    the work of one investigator reasoning toward a line of
    inquiry, not pronouncing a verdict.

A Hypothesis–Evidence draft fails when any of:
  * A Hypothesis paraphrases the factual record without a causal
    move.
  * A Hypothesis introduces mechanisms, attributions, or
    organizational claims the record cannot warrant.
  * A Hypothesis carries a probable-cause stance, writing as though
    the determination is already made.
  * Evidence is generic ("similar accidents in this industry
    suggest ..."), speculative ("one might expect ..."), or
    list-shaped enumeration without anchoring in record facts or
    absences.
  * Evidence omits the additional evidence the investigator would
    need to seek to advance the line of inquiry.
  * The factual record has notable silences but every Hypothesis
    anchors only on documented facts; the set is incomplete because
    no Hypothesis surfaces the diagnostic weight of absences.
  * {cardinality_audit_failure_clause}
  * The voice breaks — slipping into bullet-list, board-finding, or
    safety-report-summary registers that do not continue the
    Context's investigator voice.

When the draft fails, write actionable revision instructions: if the
problem lies in a single pair, name its index; otherwise name the
specific Hypothesis or Evidence passage that needs change. The
instruction must be specific enough that the Forge can execute it
without re-deriving the diagnosis; vague quality nudges ("strengthen
the hypothesis", "make the evidence more concrete") are not
acceptable.
"""

DOMAIN_HYPOTHESIS_AUDIT_ROLES: dict[str, str] = {
    "financial_analysis": _FINANCIAL_ANALYSIS_HYPOTHESIS_AUDIT_ROLE,
    "it_operations": _IT_OPERATIONS_HYPOTHESIS_AUDIT_ROLE,
    "safety_investigation": _SAFETY_INVESTIGATION_HYPOTHESIS_AUDIT_ROLE,
}

_RESEARCH_AUDIT_CARDINALITY = (
    "Exactly one Hypothesis is present, with forward-looking Evidence."
)
_RESEARCH_AUDIT_FAILURE = (
    "More than one Hypothesis is present, or the Evidence is retrospective "
    "rather than describing how the Hypothesis would be tested."
)
_REAL_WORLD_AUDIT_CARDINALITY = (
    "A small set of genuinely distinct Hypotheses is present (typically "
    "two to four), each with an Evidence package grounded in the Context "
    "and a non-empty category tag that names the analytical lane the "
    "Hypothesis represents. Where the case provides a category menu, "
    "labels are drawn from the menu by default; off-menu labels are "
    "accepted only when no menu entry fits the Hypothesis. Categories "
    "meaningfully distinguish hypotheses that occupy different analytical "
    "lanes."
)
_REAL_WORLD_AUDIT_FAILURE = (
    "Hypotheses are redundant or near-duplicates, the set is so large "
    "that low-quality entries dilute the high-quality ones, category "
    "tags are missing or vague, the tags fail to distinguish hypotheses "
    "that genuinely sit in different analytical lanes (or conversely "
    "collapse separable lanes into one tag), or off-menu labels are used "
    "where a menu entry would have fit."
)


# ---- instruction assembly ----

def principles_for(domain: DomainConfig) -> str:
    """Per-domain principle if defined; else the track-level default (research only)."""
    if domain.name in DOMAIN_PRINCIPLES:
        return DOMAIN_PRINCIPLES[domain.name]
    if domain.track in TRACK_PRINCIPLES:
        return TRACK_PRINCIPLES[domain.track]
    raise KeyError(
        f"No principle defined for domain {domain.name!r} (track {domain.track!r}). "
        f"Real-world domains require an explicit DOMAIN_PRINCIPLES entry; "
        f"the shared real_world fallback was removed because each real-world "
        f"domain needs principle text tailored to its source structure."
    )


def category_block_for(domain: DomainConfig) -> str:
    """Category section (menu-driven if labels provided, else free vocabulary)."""
    if domain.category_labels:
        menu = "\n".join(f"  - {label}" for label in domain.category_labels)
        return _CATEGORY_WITH_MENU.format(category_menu=menu)
    return _CATEGORY_NO_MENU


def _format_role(template: str, *, domain: DomainConfig, **extra: str) -> str:
    profile = domain_profile(domain)
    return template.format(**profile, **extra)


def _instructions(role_body: str) -> str:
    return f"{BENCHMARK_OVERVIEW}\n\n{role_body}".strip() + "\n"


def _context_forge_role(domain: DomainConfig) -> str:
    """Per-domain override if present, else the shared default."""
    return DOMAIN_CONTEXT_FORGE_ROLES.get(domain.name) or _CONTEXT_FORGE_ROLE


def _context_audit_role(domain: DomainConfig) -> str:
    """Per-domain override if present, else the shared default."""
    return DOMAIN_CONTEXT_AUDIT_ROLES.get(domain.name) or _CONTEXT_AUDIT_ROLE


def _hypothesis_forge_role(domain: DomainConfig) -> str:
    """Per-domain override if present, else the shared default."""
    return DOMAIN_HYPOTHESIS_FORGE_ROLES.get(domain.name) or _HYPOTHESIS_FORGE_ROLE


def _hypothesis_audit_role(domain: DomainConfig) -> str:
    """Per-domain override if present, else the shared default."""
    return DOMAIN_HYPOTHESIS_AUDIT_ROLES.get(domain.name) or _HYPOTHESIS_AUDIT_ROLE


def _real_world_cardinality(domain: DomainConfig) -> str:
    """Real-world cardinality clause; injects category menu when domain provides labels."""
    return _REAL_WORLD_CARDINALITY_BASE + "\n" + category_block_for(domain)


def context_forge_instructions(domain: DomainConfig) -> str:
    """Full instruction string for the Context Forge agent."""
    role = _format_role(_context_forge_role(domain), domain=domain)
    return _instructions(principles_for(domain) + "\n" + role)


def context_audit_instructions(domain: DomainConfig) -> str:
    """Full instruction string for the Context Audit agent."""
    role = _format_role(_context_audit_role(domain), domain=domain)
    return _instructions(principles_for(domain) + "\n" + role)


def hypothesis_forge_instructions(domain: DomainConfig) -> str:
    """Full instruction string for the Hypothesis Forge agent."""
    if domain.multi_hypothesis:
        cardinality_block = _real_world_cardinality(domain)
    else:
        cardinality_block = _RESEARCH_CARDINALITY
    role = _format_role(
        _hypothesis_forge_role(domain),
        domain=domain,
        hypothesis_cardinality_block=cardinality_block.strip(),
    )
    return _instructions(principles_for(domain) + "\n" + role)


def hypothesis_audit_instructions(domain: DomainConfig) -> str:
    """Full instruction string for the Hypothesis Audit agent."""
    if domain.multi_hypothesis:
        clause = _REAL_WORLD_AUDIT_CARDINALITY
        failure_clause = _REAL_WORLD_AUDIT_FAILURE
    else:
        clause = _RESEARCH_AUDIT_CARDINALITY
        failure_clause = _RESEARCH_AUDIT_FAILURE
    role = _format_role(
        _hypothesis_audit_role(domain),
        domain=domain,
        cardinality_audit_clause=clause,
        cardinality_audit_failure_clause=failure_clause,
    )
    return _instructions(principles_for(domain) + "\n" + role)


# ---- per-call user prompts ----

def _source_block(record: SourceRecord) -> str:
    """Render the source descriptor for the Forge's view."""
    lines = [f"Source id: {record.id}"]
    if record.title:
        lines.append(f"Title: {record.title}")
    if record.file:
        lines.append(f"Local file: {record.file}")
    if record.url:
        lines.append(f"URL: {record.url}")
    metadata = dict(record.metadata or {})
    analysis_urls = metadata.pop("analysis_urls", None)
    if metadata:
        lines.append(f"Metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}")
    if analysis_urls:
        lines.append("Analysis URLs (consult per your instructions):")
        for url in analysis_urls:
            lines.append(f"  - {url}")
    return "\n".join(lines)


def context_forge_initial_prompt(record: SourceRecord) -> str:
    """First-round user prompt for the Context Forge."""
    return (
        "Draft the Context for this benchmark case. Read the source materials "
        "below carefully, then produce a single coherent Context that meets the "
        "standard described in your instructions.\n\n"
        f"{_source_block(record)}\n\n"
        "Return the Context as a single field. Length should be whatever the "
        "situation needs to be properly briefed; err on the side of depth and "
        "density rather than brevity."
    )


def hypothesis_forge_initial_prompt(context: str, record: SourceRecord) -> str:
    """First-round user prompt for the Hypothesis Forge."""
    return (
        "Write the Hypothesis side of this benchmark case. The Context below "
        "has already been audited; treat it as the situation a careful reader "
        "is sitting with. Source-consultation rules — what you may, must, or "
        "must not consult beyond the Context — are defined by your "
        "instructions; follow them strictly.\n\n"
        f"{_source_block(record)}\n\n"
        "=== Context ===\n"
        f"{context}\n"
        "=== End Context ===\n\n"
        "Produce the Hypothesis–Evidence output in the schema your "
        "instructions describe. Keep the voice continuous with the Context."
    )


def _format_audit_issues(issues: list[AuditIssue]) -> str:
    """Render an audit's issues as a numbered revision brief."""
    if not issues:
        return "(no specific issues recorded)"
    lines = []
    for i, issue in enumerate(issues, 1):
        target = f" [target_pair={issue.target_pair}]" if issue.target_pair is not None else ""
        lines.append(
            f"{i}. Problem{target}: {issue.problem}\n"
            f"   Why it matters: {issue.why_it_matters}\n"
            f"   Revision instruction: {issue.revision_instruction}"
        )
    return "\n".join(lines)


def _revision_escalation(round_number: int) -> str:
    """Extra guidance appended when the Forge has been revising for 2+ rounds."""
    if round_number < 3:
        return ""
    return (
        f"\n\nThis is revision round {round_number}. If the audit's findings "
        "overlap with findings from prior rounds on the same passage, the "
        "underlying structure likely needs rebuilding rather than further "
        "micro-edits — consider whether a paragraph, section, or Hypothesis "
        "needs full replacement instead of patching."
    )


def context_revision_prompt(audit: AuditResult, *, round_number: int = 2) -> str:
    """Follow-up prompt to the Context Forge after a failing audit."""
    return (
        "Your previous Context draft did not pass audit. Rework it according "
        "to the audit's findings below; preserve what is working and revise "
        "only the parts the audit calls out, while keeping the Context coherent "
        "as a whole.\n\n"
        f"Audit summary: {audit.summary}\n\n"
        "Audit findings:\n"
        f"{_format_audit_issues(audit.issues)}\n\n"
        "Return the full revised Context."
        f"{_revision_escalation(round_number)}"
    )


def hypothesis_revision_prompt(audit: AuditResult, *, round_number: int = 2) -> str:
    """Follow-up prompt to the Hypothesis Forge after a failing audit."""
    return (
        "Your previous Hypothesis draft did not pass audit. Rework it according "
        "to the audit's findings below; address only the specific pairs or "
        "passages the audit calls out, and keep the rest stable so the case "
        "remains coherent.\n\n"
        f"Audit summary: {audit.summary}\n\n"
        "Audit findings:\n"
        f"{_format_audit_issues(audit.issues)}\n\n"
        "Return the full revised Hypothesis output."
        f"{_revision_escalation(round_number)}"
    )


def context_audit_prompt(context: str, record: SourceRecord) -> str:
    """User prompt to the Context Auditor."""
    return (
        "Audit the Context draft below against the benchmark's standard. "
        "Decide whether it passes; if not, write actionable revision "
        "instructions targeted at the specific passages that fall short.\n\n"
        f"{_source_block(record)}\n\n"
        "=== Context draft ===\n"
        f"{context}\n"
        "=== End draft ===\n\n"
        "Return your verdict in the audit schema: a pass/fail decision, a brief "
        "summary, and (when failing) one or more concrete issues with revision "
        "instructions."
    )


def hypothesis_audit_prompt(
    context: str,
    forge_output: dict[str, Any],
    record: SourceRecord,
) -> str:
    """User prompt to the Hypothesis Auditor."""
    return (
        "Audit the Hypothesis draft below against the benchmark's standard, "
        "using the (already audited) Context as the reference for what is "
        "warranted. Decide whether it passes; if not, write actionable "
        "revision instructions for the specific Hypothesis(es) or Evidence "
        "passage(s) that fall short. When the draft contains multiple "
        "Hypotheses, use target_pair to point at the offending one (1-indexed).\n\n"
        f"{_source_block(record)}\n\n"
        "=== Context (passed audit) ===\n"
        f"{context}\n"
        "=== End Context ===\n\n"
        "=== Hypothesis draft ===\n"
        f"{json.dumps(forge_output, ensure_ascii=False, indent=2)}\n"
        "=== End draft ===\n\n"
        "Return your verdict in the audit schema."
    )
