"""
The orchestrator.

This is the one piece that actually deserves the word "agent": it decides,
per candidate/job pair, how far to take the pipeline (prefilter -> LLM
score -> grounding check -> approval queue) and stops early wherever the
evidence doesn't support going further -- rather than a fixed script that
always does the same fixed sequence of steps to completion.

Two entry points exist because the scenario has two real triggers, and
the original single-shot prototype only implements one of them:

  * handle_rejection_event(candidate_id)
        Fires when BambooHR marks an applicant Rejected. Checks that one
        candidate against every currently open job.

  * handle_new_job_event(job_id)
        Fires when a new role opens. Sweeps every rejected candidate
        already sitting in the talent pipeline against that one new job.
        This is the half of the problem a point-in-time-only design
        (both the original prototype and, by its own description,
        Second Ascent) cannot solve: a candidate rejected in January
        for a Facilitation role has no way to surface for a Program
        Officer role that opens in March, because nothing ever looks
        at them again.

Both entry points funnel into the same `_evaluate_pair`, so the matching
logic, grounding, and approval-queue behavior are identical regardless of
which direction triggered the check -- there is exactly one matching
pipeline, not two.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from src import db
from src.models import Candidate, Job, MatchResult
from src.tools import email_draft, grounding_check, llm_match, prefilter

AUTO_FLAG_THRESHOLD = 60.0  # final_score >= this AND grounding passes -> queued for recruiter approval


def _row_to_candidate(row) -> Candidate:
    return Candidate(
        candidate_id=row["candidate_id"],
        first_name=row["first_name"],
        last_name=row["last_name"],
        email=row["email"],
        phone=row["phone"],
        applied_role=row["applied_role"],
        applied_function=row["applied_function"],
        applied_level=row["applied_level"],
        years_of_experience=row["years_of_experience"],
        location=row["location"],
        key_skills=[s.strip() for s in (row["key_skills"] or "").split(",") if s.strip()],
        secondary_interest_functions=[
            s.strip() for s in (row["secondary_interest_functions"] or "").split(",") if s.strip()
        ],
        resume_summary=row["resume_summary"],
        rejection_reason=row["rejection_reason"],
        application_date=row["application_date"],
        rejection_date=row["rejection_date"],
        status=row["status"],
    )


def _row_to_job(row) -> Job:
    return Job(
        job_id=row["job_id"],
        title=row["title"],
        function=row["function"],
        level=row["level"],
        hiring_manager_name=row["hiring_manager_name"],
        hiring_manager_email=row["hiring_manager_email"],
        required_skills=[s.strip() for s in (row["required_skills"] or "").split(",") if s.strip()],
        min_years_experience=row["min_years_experience"],
        description=row["description"],
        status=row["status"],
        opened_date=row["opened_date"],
    )


def _already_evaluated(conn, candidate_id: str, job_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM matches WHERE candidate_id = ? AND job_id = ?",
        (candidate_id, job_id),
    ).fetchone()
    return row is not None


def _persist_match(conn, match: MatchResult) -> int:
    cursor = conn.execute(
        """
        INSERT INTO matches (
            candidate_id, job_id, prefilter_score, prefilter_reasons,
            llm_score, matched_skills, missing_skills, explanation,
            grounding_passed, grounding_notes, final_score, stage, source_event
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(candidate_id, job_id) DO UPDATE SET
            prefilter_score=excluded.prefilter_score,
            prefilter_reasons=excluded.prefilter_reasons,
            llm_score=excluded.llm_score,
            matched_skills=excluded.matched_skills,
            missing_skills=excluded.missing_skills,
            explanation=excluded.explanation,
            grounding_passed=excluded.grounding_passed,
            grounding_notes=excluded.grounding_notes,
            final_score=excluded.final_score,
            stage=excluded.stage,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            match.candidate_id,
            match.job_id,
            match.prefilter_score,
            json.dumps(match.prefilter_reasons),
            match.llm_score,
            json.dumps(match.matched_skills),
            json.dumps(match.missing_skills),
            match.explanation,
            int(match.grounding_passed) if match.grounding_passed is not None else None,
            json.dumps(match.grounding_notes),
            match.final_score,
            match.stage,
            match.source_event,
        ),
    )
    if cursor.lastrowid:
        return cursor.lastrowid
    return conn.execute(
        "SELECT match_id FROM matches WHERE candidate_id = ? AND job_id = ?",
        (match.candidate_id, match.job_id),
    ).fetchone()["match_id"]


def _evaluate_pair(conn, candidate: Candidate, job: Job, source_event: str) -> MatchResult | None:
    if _already_evaluated(conn, candidate.candidate_id, job.job_id):
        return None  # dedup: never re-score a pair the agent has already looked at

    prefilter_score, prefilter_reasons = prefilter.score_pair(candidate, job)
    if prefilter_score < prefilter.MIN_PREFILTER_SCORE:
        return None  # not worth an LLM call; not even worth a "no_action" row

    llm_result = llm_match.score_candidate_against_job(candidate, job, prefilter_score)
    grounding = grounding_check.check_grounding(candidate, llm_result.matched_skills)

    # Final score blends structural fit and semantic fit, but is capped
    # by grounding: a claim-heavy, evidence-light result cannot reach the
    # auto-flag threshold no matter how confident the LLM sounded.
    final_score = llm_result.match_score
    if not grounding.passed:
        final_score = min(final_score, AUTO_FLAG_THRESHOLD - 1)

    stage = "queued_for_approval" if final_score >= AUTO_FLAG_THRESHOLD else "below_threshold"

    match = MatchResult(
        candidate_id=candidate.candidate_id,
        job_id=job.job_id,
        prefilter_score=prefilter_score,
        prefilter_reasons=prefilter_reasons,
        llm_score=llm_result.match_score,
        matched_skills=grounding.verified_skills,
        missing_skills=llm_result.missing_skills,
        explanation=llm_result.explanation
        + (f" [grounding: {' '.join(grounding.notes)}]" if grounding.notes else ""),
        grounding_passed=grounding.passed,
        grounding_notes=grounding.notes,
        final_score=final_score,
        stage=stage,
        source_event=source_event,
    )
    match_id = _persist_match(conn, match)
    match.match_id = match_id  # type: ignore[attr-defined]
    return match


def handle_rejection_event(candidate_id: str) -> list[MatchResult]:
    """Trigger 1: a candidate was just rejected. Check them against every open job."""
    results = []
    with db.connect() as conn:
        candidate_row = conn.execute(
            "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if candidate_row is None:
            raise ValueError(f"Unknown candidate_id: {candidate_id}")
        candidate = _row_to_candidate(candidate_row)

        job_rows = conn.execute("SELECT * FROM jobs WHERE status = 'open'").fetchall()
        for job_row in job_rows:
            job = _row_to_job(job_row)
            match = _evaluate_pair(conn, candidate, job, source_event="candidate_rejected")
            if match:
                results.append(match)
    return results


def handle_new_job_event(job_id: str) -> list[MatchResult]:
    """Trigger 2: a new job just opened. Sweep the existing talent pipeline for it."""
    results = []
    with db.connect() as conn:
        job_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if job_row is None:
            raise ValueError(f"Unknown job_id: {job_id}")
        job = _row_to_job(job_row)

        candidate_rows = conn.execute(
            "SELECT * FROM candidates WHERE status = 'Rejected'"
        ).fetchall()
        for candidate_row in candidate_rows:
            candidate = _row_to_candidate(candidate_row)
            match = _evaluate_pair(conn, candidate, job, source_event="job_opened")
            if match:
                results.append(match)
    return results


def approve_match(match_id: int, decided_by: str = "recruitment_officer", notes: str = "") -> str:
    """Human approval gate: nothing is emailed until this is called."""
    with db.connect() as conn:
        match_row = conn.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,)).fetchone()
        if match_row is None:
            raise ValueError(f"Unknown match_id: {match_id}")
        if match_row["stage"] != "queued_for_approval":
            raise ValueError(
                f"Match {match_id} is in stage '{match_row['stage']}', not "
                f"'queued_for_approval' -- nothing to approve."
            )

        candidate = _row_to_candidate(
            conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?", (match_row["candidate_id"],)
            ).fetchone()
        )
        job = _row_to_job(
            conn.execute("SELECT * FROM jobs WHERE job_id = ?", (match_row["job_id"],)).fetchone()
        )

        email = email_draft.draft_flag_email(
            candidate=candidate,
            job=job,
            match_score=match_row["final_score"],
            matched_skills=json.loads(match_row["matched_skills"]),
            explanation=match_row["explanation"],
            source_event=match_row["source_event"],
        )

        from src.tools import email_send

        filepath = email_send.send_email(email, match_id)

        conn.execute(
            "INSERT INTO decisions (match_id, decision, decided_by, notes) VALUES (?, 'approved', ?, ?)",
            (match_id, decided_by, notes),
        )
        conn.execute(
            "UPDATE matches SET stage = 'sent', updated_at = CURRENT_TIMESTAMP WHERE match_id = ?",
            (match_id,),
        )
        return str(filepath)


def reject_match(match_id: int, decided_by: str = "recruitment_officer", notes: str = "") -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO decisions (match_id, decision, decided_by, notes) VALUES (?, 'rejected', ?, ?)",
            (match_id, decided_by, notes),
        )
        conn.execute(
            "UPDATE matches SET stage = 'recruiter_rejected', updated_at = CURRENT_TIMESTAMP WHERE match_id = ?",
            (match_id,),
        )
