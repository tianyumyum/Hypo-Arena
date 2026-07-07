---
name: chronology
description: >
  Chronology and Timeline Analysis. Decomposes evidence into a strict
  chronological sequence, then examines time intervals, gaps, anomalous
  speeds, and temporal patterns to reveal hidden causal connections and
  missing evidence. Best suited for investigation domains (NTSB, CSB,
  complex sequenced events) where the sequence and timing of events is critical
  to understanding causation.
---

## Analytical Framework: Chronology and Timeline Analysis

Apply chronological decomposition to generate hypotheses from the case
context. This method uses the timeline as a diagnostic instrument — not
merely to organize events, but to reveal what the timing itself tells us
about causation.

### Core Concept

A timeline is more than a list of events. The INTERVALS between events,
the GAPS where expected events are absent, and the SPEED at which events
unfold all carry diagnostic information. Long gaps may indicate missing
evidence or deliberate concealment. Unexpectedly rapid sequences may
indicate cascading failures. Events that should have occurred but didn't
(the "dog that didn't bark") can be more informative than events that did.

### Procedure

1. **Extract and sequence all time-stamped events**: From the context,
   identify every event, action, measurement, or condition that has a
   timestamp or temporal reference. Arrange them in strict chronological
   order. Include:
   - Actions taken by operators/personnel
   - System state changes (alarms, readings, mode changes)
   - Environmental conditions at specific times
   - Communications and decisions

2. **Analyze intervals**: For each pair of consecutive events, examine
   the time interval:
   - **Unexpectedly long intervals**: Why did nothing happen during this
     period? Was there a delayed response? Was evidence not collected?
     Was something happening that is not documented?
   - **Unexpectedly short intervals**: Does the rapid succession suggest
     a cascading failure? Were events actually sequential or did they
     overlap? Could the rapid pace have overwhelmed human response
     capability?
   - **Expected intervals**: Do the intervals match normal operational
     timing? If so, this confirms the system was operating as designed
     during that period.

3. **Identify temporal gaps**: Look for periods where:
   - Evidence is absent but should exist (missing data, unrecorded
     actions, communication blackouts)
   - An expected action or check did not occur (a "dog that didn't bark")
   - The analyst's attention jumps forward — what happened in between?

4. **Map concurrent activities**: Identify events that occurred
   simultaneously or in close temporal proximity. Ask:
   - Were parallel activities coordinated or independent?
   - Could one activity have distracted from or interfered with another?
   - Do concurrent events suggest a common trigger?

5. **Identify the critical transition**: Find the moment in the timeline
   where the situation shifted from recoverable to unrecoverable. What
   was the last point at which intervention could have changed the
   outcome? What was happening (or not happening) at that moment?

6. **Generate timeline-derived hypotheses**: Based on the temporal analysis:
   - What does the timeline's structure (intervals, gaps, speed) suggest
     about causation that is not obvious from the events alone?
   - What hypotheses are supported by the temporal patterns?
   - What hypotheses are contradicted by the timing?

### Output Mapping

Translate the methodology's analytical output into the Hypothesis–Evidence
pairs your domain instructions specify; cardinality, category labels
(where applicable), and Evidence-anchoring rules are defined there. Use
the methodology's outputs as analytical inputs, not as output structure
— do not surface internal scaffolding (matrices, scoring tables,
step-by-step procedure, dimension lists) in the final output.

### Quality Criteria

- Events must be sequenced with specific timestamps or temporal
  references from the context, not invented.
- Interval analysis must explicitly address at least one unexpected
  gap and one unexpected rapid sequence (if present in the evidence).
- The critical transition point must be identified with reasoning for
  why that specific moment was the inflection point.
- Verification must use temporal evidence (timestamps, durations,
  sequences) as a primary form of support.
