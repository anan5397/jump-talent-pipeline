"""
Deterministic, zero-cost, zero-latency first pass.

The single-shot prototype sent every candidate to the LLM against *all*
open jobs, every time. That is 100% of pairs paying LLM cost and latency
even though most pairs are obviously irrelevant (a Finance Officer
applicant is never a plausible Outdoor Instructor match).

This tool scores plain structural fit -- function match, level distance,
years of experience vs. the role's minimum, and secondary interests --
using no external calls at all. Only pairs that clear MIN_PREFILTER_SCORE
get passed on to the LLM stage. This is the kind of cheap-before-expensive
staging a real agent should do on its own, and it is fully unit-testable
without an API key (see tests/test_prefilter.py).
"""

from __future__ import annotations

from src.models import LEVEL_RANK, Candidate, Job

MIN_PREFILTER_SCORE = 0.35

# A rejection reason that has nothing to do with whether someone can do a
# *different* job (notice period, budget, overqualified, relocation,
# internal transfer, lost to a more specialized competing candidate...)
# should not suppress re-surfacing them elsewhere. A rejection reason
# that says the candidate didn't have the skills/experience is real
# signal and should not be silently forgotten just because a different
# job title is now in play. Keyword-based on purpose -- simple, auditable,
# and doesn't require a model call just to classify a sentence.
NON_CAPABILITY_REJECTION_KEYWORDS = [
    "notice period", "overqualified", "budget", "salary", "relocat",
    "withdr", "filled internally", "frozen", "competing offer",
    "availability", "remote", "on-site", "shifted", "internal transfer",
    "another candidate", "specialized", "native fluency",
]
CAPABILITY_GAP_KEYWORDS = [
    "lacked", "lack of", "insufficient", "did not have", "does not have",
    "minimum required", "gap in", "missing", "not enough experience",
    "below the", "did not meet",
]


def classify_rejection_reason(rejection_reason: str) -> str:
    """
    Classify a rejection as 'non_capability', 'capability_gap', or
    'unknown' (the neutral default when neither keyword set matches --
    unclassified reasons get partial, not full, re-engagement credit).
    """
    text = (rejection_reason or "").lower()
    if any(keyword in text for keyword in CAPABILITY_GAP_KEYWORDS):
        return "capability_gap"
    if any(keyword in text for keyword in NON_CAPABILITY_REJECTION_KEYWORDS):
        return "non_capability"
    return "unknown"


def score_pair(candidate: Candidate, job: Job) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    # Function match is the strongest signal. A secondary-interest match
    # counts, just for less, because the candidate self-reported it.
    if candidate.applied_function.strip().lower() == job.function.strip().lower():
        score += 0.40
        reasons.append(f"Applied function matches job function ({job.function}).")
    elif job.function.strip().lower() in [
        f.strip().lower() for f in candidate.secondary_interest_functions
    ]:
        score += 0.22
        reasons.append(f"Candidate listed {job.function} as a secondary interest.")
    else:
        reasons.append(f"No function overlap with {job.function}.")

    # Level distance: an exact level match is ideal; adjacent levels
    # (e.g. Manager applicant -> Officer opening) are still plausible;
    # anything more than one band apart is not.
    candidate_rank = LEVEL_RANK.get(candidate.applied_level)
    job_rank = LEVEL_RANK.get(job.level)
    if candidate_rank is not None and job_rank is not None:
        distance = abs(candidate_rank - job_rank)
        if distance == 0:
            score += 0.20
            reasons.append("Same seniority level.")
        elif distance == 1:
            score += 0.10
            reasons.append("Adjacent seniority level.")
        else:
            reasons.append("Seniority level too far apart.")

    # Experience floor.
    if candidate.years_of_experience >= job.min_years_experience:
        score += 0.10
        reasons.append(
            f"{candidate.years_of_experience}y experience meets the "
            f"{job.min_years_experience}y minimum."
        )
    else:
        reasons.append(
            f"{candidate.years_of_experience}y experience is below the "
            f"{job.min_years_experience}y minimum."
        )

    # Skill overlap (cheap keyword overlap; the LLM stage does the
    # nuanced version of this on the pairs that make it through).
    candidate_skills = {s.strip().lower() for s in candidate.key_skills}
    job_skills = {s.strip().lower() for s in job.required_skills}
    overlap = candidate_skills & job_skills
    if overlap:
        bonus = min(0.20, 0.06 * len(overlap))
        score += bonus
        reasons.append(f"Keyword skill overlap: {', '.join(sorted(overlap))}.")

    # Re-engagement: don't let a rejection for a real skills/experience
    # gap quietly vanish just because a different role is being checked;
    # do give full credit when the rejection was about something that
    # says nothing about capability (notice period, budget, relocation...).
    rejection_class = classify_rejection_reason(candidate.rejection_reason)
    if rejection_class == "non_capability":
        score += 0.10
        reasons.append("Rejected for a non-capability reason -- full re-engagement credit.")
    elif rejection_class == "capability_gap":
        reasons.append(
            "Original rejection cited a skills/experience gap -- no re-engagement credit."
        )
    else:
        score += 0.05
        reasons.append("Rejection reason doesn't clearly indicate capability either way.")

    return round(min(score, 1.0), 3), reasons


def prefilter_jobs_for_candidate(
    candidate: Candidate, jobs: list[Job], top_n: int = 3
) -> list[tuple[Job, float, list[str]]]:
    """Return up to `top_n` open jobs worth an LLM call, best first."""
    scored = [
        (job, *score_pair(candidate, job)) for job in jobs if job.status == "open"
    ]
    scored = [(job, s, r) for job, s, r in scored if s >= MIN_PREFILTER_SCORE]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_n]
