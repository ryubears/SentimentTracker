"""
Lambda entry point: pull the db from S3, run one period, push it back.

Kept deliberately thin — all the pipeline logic stays in run_period.py so the
scheduled cloud run and a local run execute the same code path.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sentiment_tracker import runtime, storage

DB_PATH = os.environ.setdefault("TRACKER_DB_PATH", "/tmp/tracker.sqlite")

def handler(event, context):
    runtime.load_secrets()
    had_db = storage.pull(DB_PATH)

    import run_period
    run_period.main(os.environ.get("TRACKER_CONFIG", "/var/task/config.yaml"))

    # Push inside the same invocation: a failure here must surface as a failed
    # run, because the period was written to a /tmp file that is about to vanish.
    pushed = storage.push(DB_PATH)
    return {"ok": True, "restored_existing_db": had_db, "persisted": pushed}
