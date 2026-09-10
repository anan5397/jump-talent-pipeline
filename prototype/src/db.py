"""
Persistent state for the talent pipeline.

This is the single biggest structural difference from a one-shot script:
a script that reads a webhook, calls an LLM, and writes one spreadsheet
row has no memory. It cannot answer "have we already flagged this pair?",
"what happened the last three times we flagged someone at this score?",
or "which rejected candidates have never been checked against role X?".

SQLite is used (not Sheets, not a dict in memory) so those questions are
just queries, and so the CLI, the webhook adapter, and a future scheduled
sweep job can all share the same state safely.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "talent_pipeline.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone TEXT,
    applied_role TEXT,
    applied_function TEXT,
    applied_level TEXT,
    years_of_experience INTEGER,
    location TEXT,
    key_skills TEXT,               -- comma-separated
    secondary_interest_functions TEXT,  -- comma-separated
    resume_summary TEXT,
    rejection_reason TEXT,
    application_date TEXT,
    rejection_date TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    title TEXT,
    function TEXT,
    level TEXT,
    hiring_manager_name TEXT,
    hiring_manager_email TEXT,
    required_skills TEXT,          -- comma-separated
    min_years_experience INTEGER,
    description TEXT,
    status TEXT,
    opened_date TEXT
);

-- One row per (candidate, job) pair the agent has ever evaluated.
-- The UNIQUE constraint is the dedup mechanism: re-running the same
-- trigger twice (or a webhook firing twice, which BambooHR-style
-- webhooks are known to do) cannot double-queue or double-email.
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    prefilter_score REAL,
    prefilter_reasons TEXT,
    llm_score REAL,
    matched_skills TEXT,
    missing_skills TEXT,
    explanation TEXT,
    grounding_passed INTEGER,
    grounding_notes TEXT,
    final_score REAL,
    stage TEXT NOT NULL,
    source_event TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id, job_id)
);

-- Every recruiter action on a queued match, kept even after the match
-- itself moves on. This is the audit trail a compliance review or an
-- EEO/bias check would ask for, and the raw material for recalibrating
-- the auto-flag threshold later (e.g. "matches scored 70-75 were
-- approved 90% of the time -> the threshold is too conservative").
CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    decision TEXT NOT NULL,        -- 'approved' | 'rejected'
    decided_by TEXT,
    notes TEXT,
    decided_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(match_id)
);
"""


@contextmanager
def connect(db_path: Path | None = None):
    # db_path defaults to None (not `= DB_PATH` directly) so this reads
    # the module attribute at *call* time, not at import time. With a
    # bound default, monkeypatching db.DB_PATH in a test (or repointing
    # it any other way) would silently have no effect on any of the many
    # `db.connect()` calls elsewhere in this codebase that don't pass a
    # path explicitly -- they'd keep using whatever DB_PATH resolved to
    # when this module was first imported.
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------
# Generic upserts, shared by every ingestion path (CSV import, JSON job
# list, and the Google Sheets bridge). One INSERT ON CONFLICT statement
# per table instead of one copy per data source, so "how does a
# candidate get into the pipeline" has a single answer regardless of
# whether it arrived via BambooHR, a CSV, or a careers-page application.
# ---------------------------------------------------------------------

def upsert_candidate(conn: sqlite3.Connection, candidate: dict) -> None:
    conn.execute(
        """
        INSERT INTO candidates (
            candidate_id, first_name, last_name, email, phone,
            applied_role, applied_function, applied_level,
            years_of_experience, location, key_skills,
            secondary_interest_functions, resume_summary,
            rejection_reason, application_date, rejection_date, status
        ) VALUES (
            :candidate_id, :first_name, :last_name, :email, :phone,
            :applied_role, :applied_function, :applied_level,
            :years_of_experience, :location, :key_skills,
            :secondary_interest_functions, :resume_summary,
            :rejection_reason, :application_date, :rejection_date, :status
        )
        ON CONFLICT(candidate_id) DO UPDATE SET
            first_name=excluded.first_name,
            last_name=excluded.last_name,
            email=excluded.email,
            phone=excluded.phone,
            applied_role=excluded.applied_role,
            applied_function=excluded.applied_function,
            applied_level=excluded.applied_level,
            years_of_experience=excluded.years_of_experience,
            location=excluded.location,
            key_skills=excluded.key_skills,
            secondary_interest_functions=excluded.secondary_interest_functions,
            resume_summary=excluded.resume_summary,
            rejection_reason=excluded.rejection_reason,
            rejection_date=excluded.rejection_date,
            status=excluded.status
        """,
        candidate,
    )


def upsert_job(conn: sqlite3.Connection, job: dict) -> None:
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, title, function, level, hiring_manager_name,
            hiring_manager_email, required_skills, min_years_experience,
            description, status, opened_date
        ) VALUES (
            :job_id, :title, :function, :level, :hiring_manager_name,
            :hiring_manager_email, :required_skills, :min_years_experience,
            :description, :status, :opened_date
        )
        ON CONFLICT(job_id) DO UPDATE SET
            title=excluded.title,
            function=excluded.function,
            level=excluded.level,
            hiring_manager_name=excluded.hiring_manager_name,
            hiring_manager_email=excluded.hiring_manager_email,
            required_skills=excluded.required_skills,
            min_years_experience=excluded.min_years_experience,
            description=excluded.description,
            status=excluded.status,
            opened_date=excluded.opened_date
        """,
        job,
    )
