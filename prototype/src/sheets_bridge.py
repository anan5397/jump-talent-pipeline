"""
Bridge between the Google Sheet the careers page writes to (see
../careers-page/) and the already-built, already-tested agent pipeline
in agent.py.

This module deliberately does no matching logic of its own -- it only
translates Sheet rows into the same Candidate/Job shape the CLI and CSV
importer already use, and calls the same handle_rejection_event /
handle_new_job_event functions. The matching pipeline behaves identically
regardless of whether a candidate arrived via BambooHR, a CSV, or a
website application; only the transport differs.

Split into two layers on purpose:
  * Pure mapping functions (`*_row_to_*`) with no gspread dependency --
    unit-tested in tests/test_sheets_bridge.py with no network or
    credentials required.
  * Thin gspread-calling functions that use those pure functions --
    exercised manually / in production, not by the offline test suite.

Run as a scheduled job (cron, Windows Task Scheduler, or a cloud
function) every few minutes:

    python -m src.sheets_bridge
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src import agent, db

JOBS_TAB = os.environ.get("SHEETS_JOBS_TAB", "Jobs")
REJECTED_TAB = os.environ.get("SHEETS_REJECTED_TAB", "Rejected")
MATCHES_TAB = os.environ.get("SHEETS_MATCHES_TAB", "Talent Matches")


# ---------------------------------------------------------------------
# Pure mapping functions -- no gspread, no network, fully unit-testable.
# ---------------------------------------------------------------------

def job_row_to_dict(row: dict) -> dict:
    """Map one row (as returned by gspread's get_all_records()) from the
    Jobs tab into the dict shape db.upsert_job() expects."""
    return {
        "job_id": row["job_id"],
        "title": row["title"],
        "function": row["function"],
        "level": row["level"],
        "hiring_manager_name": row.get("hiring_manager_name", ""),
        "hiring_manager_email": row.get("hiring_manager_email", ""),
        "required_skills": row.get("required_skills", ""),
        "min_years_experience": int(row.get("min_years_experience") or 0),
        "description": row.get("description", ""),
        "status": str(row.get("status", "")).strip() or "closed",
        "opened_date": row.get("opened_date", ""),
    }


def rejected_row_to_candidate_dict(row: dict) -> dict:
    """Map one row from the Rejected tab (written by Code.gs when HR
    marks an Applicants row as Rejected) into the dict shape
    db.upsert_candidate() expects."""
    return {
        "candidate_id": row["candidate_id"],
        "first_name": row.get("first_name", ""),
        "last_name": row.get("last_name", ""),
        "email": row.get("email", ""),
        "phone": row.get("phone", ""),
        "applied_role": row.get("applied_role", ""),
        "applied_function": row.get("applied_function", ""),
        "applied_level": row.get("applied_level", ""),
        "years_of_experience": int(row.get("years_of_experience") or 0),
        "location": row.get("location", ""),
        "key_skills": row.get("key_skills", ""),
        "secondary_interest_functions": row.get("secondary_interest_functions", ""),
        "resume_summary": row.get("resume_summary", ""),
        "rejection_reason": row.get("rejection_reason", ""),
        "application_date": row.get("submitted_at", ""),
        "rejection_date": row.get("decided_at", ""),
        "status": "Rejected",
    }


def is_row_unsynced(row: dict) -> bool:
    """The Rejected tab's Synced column is FALSE/blank until this bridge
    has ingested the row -- mirrors the dedup already enforced inside
    agent.py, but at the Sheet layer too, so re-running a sync doesn't
    re-read rows it has no new work to do for."""
    value = row.get("Synced", False)
    return str(value).strip().lower() not in ("true", "1", "yes")


# ---------------------------------------------------------------------
# gspread-calling layer.
# ---------------------------------------------------------------------

def _get_worksheet(spreadsheet_name: str, tab_name: str):
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
    credentials = Credentials.from_service_account_file(creds_file, scopes=scopes)
    client = gspread.authorize(credentials)

    # Opening by URL or ID (set GOOGLE_SPREADSHEET_URL or _ID in .env) is
    # preferred over opening by title -- it can't be broken by a rename,
    # and it's unambiguous even if multiple sheets share a name. Title
    # (spreadsheet_name) is kept as a fallback so this still works with
    # no extra configuration beyond the sheet's actual name.
    spreadsheet_url = os.environ.get("GOOGLE_SPREADSHEET_URL")
    spreadsheet_id = os.environ.get("GOOGLE_SPREADSHEET_ID")
    if spreadsheet_url:
        spreadsheet = client.open_by_url(spreadsheet_url)
    elif spreadsheet_id:
        spreadsheet = client.open_by_key(spreadsheet_id)
    else:
        spreadsheet = client.open(spreadsheet_name)

    return spreadsheet.worksheet(tab_name)


def sync_jobs(spreadsheet_name: str) -> int:
    """Pull the Jobs tab into local state, then re-check every open job
    against the entire historical talent pool. Cheap to call repeatedly:
    agent.py's dedup means an already-evaluated pair is a no-op."""
    worksheet = _get_worksheet(spreadsheet_name, JOBS_TAB)
    rows = worksheet.get_all_records()

    swept = 0
    with db.connect() as conn:
        for row in rows:
            db.upsert_job(conn, job_row_to_dict(row))

    for row in rows:
        if str(row.get("status", "")).strip().lower() == "open":
            agent.handle_new_job_event(row["job_id"])
            swept += 1
    return swept


def sync_rejected(spreadsheet_name: str) -> int:
    """Pull newly-rejected applicants and check each against every open
    job. Marks each row Synced in the Sheet once ingested."""
    worksheet = _get_worksheet(spreadsheet_name, REJECTED_TAB)
    rows = worksheet.get_all_records()
    header = worksheet.row_values(1)
    synced_col = header.index("Synced") + 1 if "Synced" in header else None

    unsynced = [
        (sheet_row_number, row)
        for sheet_row_number, row in enumerate(rows, start=2)  # row 1 is the header
        if is_row_unsynced(row)
    ]

    # Write every unsynced candidate first, in one transaction that fully
    # commits before any matching runs. agent.handle_rejection_event()
    # opens its own separate database connection -- if it ran inside the
    # same `with db.connect()` block as the upsert above, it would be
    # looking up a candidate row that technically exists but hasn't been
    # committed yet, and would raise "Unknown candidate_id".
    with db.connect() as conn:
        for _, row in unsynced:
            db.upsert_candidate(conn, rejected_row_to_candidate_dict(row))

    processed = 0
    for sheet_row_number, row in unsynced:
        agent.handle_rejection_event(row["candidate_id"])
        if synced_col:
            worksheet.update_cell(sheet_row_number, synced_col, True)
        processed += 1
    return processed


def push_pending_matches(spreadsheet_name: str) -> int:
    """Write every match currently queued_for_approval into the Talent
    Matches tab, so HR reviews AI suggestions the same way they already
    review applicants -- inside the spreadsheet, using the same
    Approval? Yes/No convention as their existing candidate export."""
    worksheet = _get_worksheet(spreadsheet_name, MATCHES_TAB)
    existing_rows = worksheet.get_all_records()
    already_pushed = {str(r.get("match_id")) for r in existing_rows}

    pushed = 0
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT m.match_id, c.first_name, c.last_name, c.email,
                   j.title, m.final_score, m.matched_skills, m.explanation,
                   m.source_event
            FROM matches m
            JOIN candidates c ON c.candidate_id = m.candidate_id
            JOIN jobs j ON j.job_id = m.job_id
            WHERE m.stage = 'queued_for_approval'
            """
        ).fetchall()

    for row in rows:
        if str(row["match_id"]) in already_pushed:
            continue
        worksheet.append_row(
            [
                row["match_id"],
                f"{row['first_name']} {row['last_name']}",
                row["email"],
                row["title"],
                row["final_score"],
                row["matched_skills"],
                row["explanation"],
                row["source_event"],
                "",  # Approval? -- left blank for HR to fill in
                "",  # Processed
            ]
        )
        pushed += 1
    return pushed


def pull_approval_decisions(spreadsheet_name: str) -> int:
    """Read HR's Yes/No entries in the Approval? column and act on them
    -- this is what actually triggers the (mocked) hiring-manager email
    via agent.approve_match()."""
    worksheet = _get_worksheet(spreadsheet_name, MATCHES_TAB)
    rows = worksheet.get_all_records()
    header = worksheet.row_values(1)
    processed_col = header.index("Processed") + 1 if "Processed" in header else None

    actioned = 0
    for sheet_row_number, row in enumerate(rows, start=2):
        if str(row.get("Processed", "")).strip().lower() in ("true", "yes", "1"):
            continue
        approval = str(row.get("Approval?", "")).strip().lower()
        if approval == "yes":
            agent.approve_match(int(row["match_id"]), decided_by="hr_via_sheet")
            actioned += 1
        elif approval == "no":
            agent.reject_match(int(row["match_id"]), decided_by="hr_via_sheet")
            actioned += 1
        else:
            continue  # still awaiting HR's decision
        if processed_col:
            worksheet.update_cell(sheet_row_number, processed_col, True)
    return actioned


def run_sync_cycle(spreadsheet_name: str | None = None) -> dict:
    spreadsheet_name = spreadsheet_name or os.environ.get(
        "GOOGLE_SPREADSHEET_NAME", "JUMP Talent Pipeline"
    )
    db.init_db()
    return {
        "jobs_swept": sync_jobs(spreadsheet_name),
        "rejected_ingested": sync_rejected(spreadsheet_name),
        "matches_pushed_to_sheet": push_pending_matches(spreadsheet_name),
        "approvals_actioned": pull_approval_decisions(spreadsheet_name),
    }


if __name__ == "__main__":
    import argparse
    import time

    cli = argparse.ArgumentParser(description="Sync the Google Sheet with the talent pipeline agent.")
    cli.add_argument(
        "--watch", action="store_true",
        help="keep running, re-syncing on an interval instead of syncing once and exiting",
    )
    cli.add_argument("--interval", type=int, default=30, help="seconds between syncs in --watch mode (default 30)")
    parsed = cli.parse_args()

    if not parsed.watch:
        print(run_sync_cycle())
    else:
        print(f"Watching the Google Sheet every {parsed.interval}s (Ctrl+C to stop)...")
        try:
            while True:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {run_sync_cycle()}")
                time.sleep(parsed.interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")
