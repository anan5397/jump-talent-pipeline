"""
Build the hiring-manager flag email.

Deliberately template-based, not LLM-generated: the content of a
recruiter-facing, hiring-manager-facing email should be predictable and
auditable word-for-word, and there is no ambiguity here an LLM needs to
resolve. Save the model calls for the steps that actually need judgement
(matching), not the steps that just need consistent formatting.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import Candidate, Job


@dataclass
class DraftEmail:
    to: str
    subject: str
    body: str


def draft_flag_email(
    candidate: Candidate,
    job: Job,
    match_score: float,
    matched_skills: list[str],
    explanation: str,
    source_event: str,
) -> DraftEmail:
    trigger_line = (
        "This candidate was just rejected from another role and looks like a "
        "possible fit for your open position."
        if source_event == "candidate_rejected"
        else "Your role was just opened, and this candidate from JUMP!'s "
        "existing talent pipeline looks like a possible fit."
    )

    subject = f"[Talent Pipeline] Possible fit for {job.title}: {candidate.full_name} ({match_score:.0f}% match)"

    body = f"""Hi {job.hiring_manager_name},

{trigger_line}

Candidate: {candidate.full_name}
Originally applied for: {candidate.applied_role} ({candidate.applied_function}, {candidate.applied_level})
Years of experience: {candidate.years_of_experience}
Location: {candidate.location}
Contact: {candidate.email} / {candidate.phone}

Suggested fit: {job.title} ({job.function}, {job.level}) -- {match_score:.0f}% match
Matched, evidence-checked skills: {", ".join(matched_skills) if matched_skills else "none confidently verified"}

Why this was flagged:
{explanation}

Original rejection reason (for context, not a reflection on this role):
{candidate.rejection_reason}

This suggestion was screened by an AI-assisted talent pipeline and reviewed by the \
recruitment officer before reaching you. It is a recommendation, not a hiring \
decision -- please review the candidate profile before proceeding.

-- JUMP! Talent Pipeline
"""
    return DraftEmail(to=job.hiring_manager_email, subject=subject, body=body)
