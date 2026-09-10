"""
Typed data structures shared across the pipeline.

Keeping these as plain dataclasses (rather than burying the shape of a
candidate/job/match inside dicts passed between functions) means every
tool in src/tools/ has a documented contract. It also means the schema
that a human reviewer sees, the schema a hiring-manager email is built
from, and the schema stored in SQLite are all the *same* object -- one
source of truth instead of three places that can quietly drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# The function x role/level matrix requested in the task brief. Kept as
# explicit constants (not inferred from free text) so "create a talent
# pipeline with these categories" is enforced, not just implied.
FUNCTIONS = [
    "Program design",
    "Operations",
    "Facilitation",
    "Marketing",
    "Business development",
    "Finance",
]

LEVELS = ["Contractor", "Officer", "Manager", "Director", "Executive"]

# Used by the prefilter tool to score how "close" two levels are
# (e.g. a Director is a plausible fit for a Manager opening; a
# Contractor applicant is not a plausible fit for an Executive opening).
LEVEL_RANK = {level: index for index, level in enumerate(LEVELS)}


@dataclass
class Candidate:
    candidate_id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    applied_role: str
    applied_function: str
    applied_level: str
    years_of_experience: int
    location: str
    key_skills: list[str]
    secondary_interest_functions: list[str]
    resume_summary: str
    rejection_reason: str
    application_date: str
    rejection_date: str
    status: str

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


@dataclass
class Job:
    job_id: str
    title: str
    function: str
    level: str
    hiring_manager_name: str
    hiring_manager_email: str
    required_skills: list[str]
    min_years_experience: int
    description: str
    status: str
    opened_date: str


@dataclass
class MatchResult:
    """
    The output of scoring one (candidate, job) pair.

    `stage` records how far the pair got through the pipeline, which is
    what makes the run auditable: a low-confidence match is not silently
    discarded, it is stored with a reason so it can be re-swept later
    (e.g. once the candidate's real record has more history, or once a
    grounding failure is fixed upstream).
    """

    candidate_id: str
    job_id: str
    prefilter_score: float
    prefilter_reasons: list[str]
    llm_score: Optional[float] = None
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    explanation: str = ""
    grounding_passed: Optional[bool] = None
    grounding_notes: list[str] = field(default_factory=list)
    final_score: Optional[float] = None
    stage: str = "prefiltered"  # prefiltered -> llm_scored -> grounded -> queued -> approved/rejected -> sent
    source_event: str = ""  # "candidate_rejected" or "job_opened" -- which trigger produced this match
