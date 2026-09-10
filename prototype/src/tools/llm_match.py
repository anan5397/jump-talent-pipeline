"""
Deep semantic scoring for the (small) set of pairs the prefilter judged
plausible.

Design choices that differ from the single-shot prototype's ai_matcher.py:

  * Only called on prefiltered pairs, not the full job list every time.
  * Uses Claude's native structured tool-calling (an input_schema the
    model must satisfy) instead of asking the model to "return valid
    JSON only, no markdown" inside a text prompt and then regex-stripping
    code fences from the response. Malformed output becomes a schema
    validation error you can catch, not a JSONDecodeError on markdown
    fences.
  * Falls back to a transparent, deterministic stub when no
    ANTHROPIC_API_KEY is configured, so the rest of the pipeline
    (grounding, approval queue, email draft, audit log) can be exercised
    and demoed end-to-end without needing credentials. The fallback is
    clearly labeled in its own output -- it never silently pretends to
    be a real model judgement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.models import Candidate, Job

MATCH_TOOL_SCHEMA = {
    "name": "record_match_score",
    "description": "Record a job-fit assessment for one candidate/job pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "match_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Conservative job-fit score, evidence-based only.",
            },
            "matched_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Skills present in the candidate profile that satisfy job requirements.",
            },
            "missing_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required skills not evidenced in the candidate profile.",
            },
            "explanation": {
                "type": "string",
                "description": "One or two sentences of evidence-based rationale.",
            },
        },
        "required": ["match_score", "matched_skills", "missing_skills", "explanation"],
    },
}

SYSTEM_PROMPT = """You are assisting a recruitment officer with talent rediscovery \
at JUMP!, an experiential education organization.

Rules:
- Use only job-relevant evidence in the candidate profile provided.
- Do not use or infer protected or sensitive characteristics.
- Do not make a final hiring or rejection decision -- you are producing \
a recommendation for human review only.
- Do not invent skills or experience that are not stated.
- Be conservative with scores.
- You must call the record_match_score tool exactly once with your assessment."""


@dataclass
class LLMMatchResult:
    match_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str
    source: str  # "claude" or "offline_stub"


def _offline_stub(candidate: Candidate, job: Job, prefilter_score: float) -> LLMMatchResult:
    """
    Deterministic stand-in used when ANTHROPIC_API_KEY is not set.

    It reuses the prefilter's keyword overlap rather than inventing a
    number, and is labeled `source="offline_stub"` end to end so nothing
    downstream (including the audit log and the recruiter-facing email)
    can mistake it for a real model judgement.
    """
    candidate_skills = {s.strip().lower() for s in candidate.key_skills}
    job_skills = {s.strip() for s in job.required_skills}
    matched = [s for s in job_skills if s.lower() in candidate_skills]
    missing = [s for s in job_skills if s.lower() not in candidate_skills]
    score = round(prefilter_score * 100)
    explanation = (
        f"[offline stub -- no LLM call made] Structural fit only: "
        f"{candidate.applied_function}/{candidate.applied_level} candidate against "
        f"{job.function}/{job.level} opening, {len(matched)}/{len(job_skills)} "
        f"required skills keyword-matched."
    )
    return LLMMatchResult(score, matched, missing, explanation, source="offline_stub")


def score_candidate_against_job(
    candidate: Candidate, job: Job, prefilter_score: float
) -> LLMMatchResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _offline_stub(candidate, job, prefilter_score)

    import anthropic  # imported lazily so the package is optional offline

    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    candidate_profile = {
        "applied_role": candidate.applied_role,
        "applied_function": candidate.applied_function,
        "applied_level": candidate.applied_level,
        "years_of_experience": candidate.years_of_experience,
        "key_skills": candidate.key_skills,
        "secondary_interest_functions": candidate.secondary_interest_functions,
        "resume_summary": candidate.resume_summary,
        "rejection_reason": candidate.rejection_reason,
    }
    job_profile = {
        "title": job.title,
        "function": job.function,
        "level": job.level,
        "required_skills": job.required_skills,
        "min_years_experience": job.min_years_experience,
        "description": job.description,
    }

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[MATCH_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "record_match_score"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Candidate profile:\n{candidate_profile}\n\n"
                    f"Job opening:\n{job_profile}"
                ),
            }
        ],
    )

    tool_use_block = next(
        block for block in response.content if block.type == "tool_use"
    )
    payload = tool_use_block.input

    return LLMMatchResult(
        match_score=float(payload["match_score"]),
        matched_skills=list(payload["matched_skills"]),
        missing_skills=list(payload["missing_skills"]),
        explanation=str(payload["explanation"]),
        source="claude",
    )
