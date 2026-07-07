"""Per-domain configuration: track, source kind, search policy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Track = Literal["research", "real_world"]


class DomainConfig(BaseModel):
    """All domain-specific knobs the pipeline needs at runtime."""

    category_labels: tuple[str, ...] | None = None
    context_search_enabled: bool
    multi_hypothesis: bool
    name: str
    source_kind: str
    track: Track


DOMAINS: dict[str, DomainConfig] = {
    "biomedical_science": DomainConfig(
        name="biomedical_science",
        track="research",
        multi_hypothesis=False,
        source_kind="paper_pdf",
        context_search_enabled=True,
    ),
    "machine_learning": DomainConfig(
        name="machine_learning",
        track="research",
        multi_hypothesis=False,
        source_kind="paper_pdf",
        context_search_enabled=True,
    ),
    "social_science": DomainConfig(
        name="social_science",
        track="research",
        multi_hypothesis=False,
        source_kind="paper_pdf",
        context_search_enabled=True,
    ),
    "financial_analysis": DomainConfig(
        name="financial_analysis",
        track="real_world",
        multi_hypothesis=True,
        source_kind="filing_pdf",
        context_search_enabled=False,
        category_labels=(
            "risk_or_exposure_signal",
            "structural_or_business_model_concern",
            "margin_or_profitability_tension",
            "demand_or_market_shift",
            "cost_or_supply_pressure",
            "capital_efficiency_or_allocation_concern",
            "accounting_or_disclosure_quality_signal",
            "governance_or_management_credibility_signal",
            "strategic_dependence_or_positioning_shift",
        ),
    ),
    "it_operations": DomainConfig(
        name="it_operations",
        track="real_world",
        multi_hypothesis=True,
        source_kind="postmortem_url",
        context_search_enabled=True,
        category_labels=(
            "failure_mechanism_or_chain",
            "architectural_or_design_vulnerability",
            "change_or_deployment_induced_fault",
            "capacity_or_load_planning_gap",
            "detection_or_observability_gap",
            "procedural_or_runbook_gap",
            "dependency_or_integration_brittleness",
            "recovery_or_remediation_breakdown",
            "latent_or_dormant_hazard",
        ),
    ),
    "safety_investigation": DomainConfig(
        name="safety_investigation",
        track="real_world",
        multi_hypothesis=True,
        source_kind="incident_report_pdf",
        context_search_enabled=False,
        category_labels=(
            "accident_mechanism_or_chain",
            "barrier_or_safeguard_inadequacy",
            "human_system_interaction_or_procedural_gap",
            "maintenance_inspection_or_design_weakness",
            "organizational_or_management_system_vulnerability",
            "regulatory_or_oversight_gap",
            "latent_or_systemic_condition",
            "external_or_environmental_trigger",
        ),
    ),
}

RESEARCH_DOMAINS = tuple(d.name for d in DOMAINS.values() if d.track == "research")
REAL_WORLD_DOMAINS = tuple(d.name for d in DOMAINS.values() if d.track == "real_world")
ALL_DOMAINS = tuple(DOMAINS)


def get_domain(name: str) -> DomainConfig:
    """Look up DomainConfig by name; raise on unknown."""
    if name not in DOMAINS:
        raise KeyError(f"Unknown domain: {name!r}. Known: {sorted(DOMAINS)}")
    return DOMAINS[name]
