"""
Load the BambooHR-style export (the same CSV used for the original
spreadsheet prototype) and the JUMP! job openings into the talent
pipeline's persistent store.

In production this step is what a BambooHR webhook / nightly export pull
would populate automatically; here it is a one-off script so the rest of
the pipeline can be run and inspected without any external credentials.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import db

PROTOTYPE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES_CSV = (
    PROTOTYPE_DIR.parent / "JUMP Recruitment Pipeline - Candidates.csv"
)
DEFAULT_JOBS_JSON = PROTOTYPE_DIR / "data" / "jobs.json"


def _split_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def import_candidates(csv_path: Path = DEFAULT_CANDIDATES_CSV) -> int:
    db.init_db()
    count = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        with db.connect() as conn:
            for row in reader:
                # Only rejected candidates enter the talent-rediscovery
                # flow -- mirrors the scenario's actual trigger.
                if row.get("status", "").strip().lower() != "rejected":
                    continue
                db.upsert_candidate(
                    conn,
                    {
                        "candidate_id": row["candidate_id"],
                        "first_name": row["first_name"],
                        "last_name": row["last_name"],
                        "email": row["email"],
                        "phone": row["phone"],
                        "applied_role": row["applied_role"],
                        "applied_function": row["applied_function"],
                        "applied_level": row["applied_level"],
                        "years_of_experience": int(row["years_of_experience"] or 0),
                        "location": row["location"],
                        "key_skills": ", ".join(_split_list(row.get("key_skills", ""))),
                        "secondary_interest_functions": ", ".join(
                            _split_list(row.get("secondary_interest_functions", ""))
                        ),
                        "resume_summary": row.get("resume_summary", ""),
                        "rejection_reason": row.get("rejection_reason", ""),
                        "application_date": row.get("application_date", ""),
                        "rejection_date": row.get("rejection_date", ""),
                        "status": row["status"],
                    },
                )
                count += 1
    return count


def import_jobs(json_path: Path = DEFAULT_JOBS_JSON) -> int:
    db.init_db()
    jobs = json.loads(json_path.read_text(encoding="utf-8"))
    with db.connect() as conn:
        for job in jobs:
            db.upsert_job(
                conn,
                {
                    "job_id": job["job_id"],
                    "title": job["title"],
                    "function": job["function"],
                    "level": job["level"],
                    "hiring_manager_name": job["hiring_manager_name"],
                    "hiring_manager_email": job["hiring_manager_email"],
                    "required_skills": ", ".join(job["required_skills"]),
                    "min_years_experience": job["min_years_experience"],
                    "description": job["description"],
                    "status": job["status"],
                    "opened_date": job["opened_date"],
                },
            )
    return len(jobs)


if __name__ == "__main__":
    n_candidates = import_candidates()
    n_jobs = import_jobs()
    print(f"Imported {n_candidates} rejected candidates and {n_jobs} open jobs "
          f"into {db.DB_PATH}")
