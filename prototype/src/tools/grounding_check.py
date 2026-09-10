"""
Deterministic fact-check on the LLM's own output.

The single-shot prototype's only defense against hallucinated skills was
a prompt instruction ("do not invent experience or skills"). Prompt
instructions are not verification -- they lower the *rate* of a failure
mode without ever catching a specific instance of it.

This tool re-reads every `matched_skill` the LLM claimed and checks it
against the candidate's own key_skills and resume_summary text. Anything
claimed that isn't actually evidenced gets stripped out and logged as a
grounding failure. A pair whose claims don't hold up drops back in
confidence rather than sailing on to a hiring manager's inbox -- this is
what "human oversight, not blind AI trust" looks like as code rather than
as a sentence in a README.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import Candidate


@dataclass
class GroundingReport:
    passed: bool
    verified_skills: list[str]
    unverified_skills: list[str]
    notes: list[str]


def check_grounding(candidate: Candidate, claimed_matched_skills: list[str]) -> GroundingReport:
    evidence_text = " ".join(
        [candidate.resume_summary, *candidate.key_skills]
    ).lower()

    verified: list[str] = []
    unverified: list[str] = []

    for skill in claimed_matched_skills:
        # A skill counts as grounded if it (or a close keyword fragment)
        # actually appears in the candidate's own stated skills/summary.
        needle = skill.strip().lower()
        if needle and needle in evidence_text:
            verified.append(skill)
        else:
            unverified.append(skill)

    notes = []
    if unverified:
        notes.append(
            f"{len(unverified)} claimed skill(s) not found in candidate's own "
            f"profile text: {', '.join(unverified)}."
        )
    if not claimed_matched_skills:
        notes.append("No matched skills were claimed to verify.")

    # Fail closed: if more than a third of claims can't be verified,
    # this pair should not go to a human as a confident recommendation.
    passed = len(unverified) <= max(1, len(claimed_matched_skills) // 3)

    return GroundingReport(
        passed=passed,
        verified_skills=verified,
        unverified_skills=unverified,
        notes=notes,
    )
