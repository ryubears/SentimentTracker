"""
Keep the SQLite file in S3 between Lambda invocations.

Lambda has no durable disk, so the database is pulled to /tmp at the start of a
run and pushed back at the end. That is sound here only because the workload is
a single writer firing once an hour: there is no concurrent access to merge.

Two guards make that assumption safe rather than merely likely:
  * the function is deployed with reserved concurrency 1, so AWS will not run
    two invocations at once, and
  * push() sends the object's ETag as an If-Match precondition, so if something
    did write in between, the upload fails loudly instead of silently
    discarding the other run's periods.

No-ops when TRACKER_S3_BUCKET is unset, so local runs are unaffected.
"""

from __future__ import annotations
from pathlib import Path
import os

def _target() -> tuple[str, str] | None:
    bucket = os.environ.get("TRACKER_S3_BUCKET")
    return (bucket, os.environ.get("TRACKER_S3_KEY", "tracker.sqlite")) if bucket else None

_etag: str | None = None

def pull(local_path: str) -> bool:
    """Download the database, remembering its ETag for the matching push.
    Returns False when there is nothing to download (first ever run)."""
    global _etag
    target = _target()
    if not target:
        return False
    import boto3, botocore
    bucket, key = target
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            _etag = None
            return False # first run: db.connect will create the schema
        raise
    _etag = obj["ETag"]
    Path(local_path).write_bytes(obj["Body"].read())
    return True

def push(local_path: str) -> bool:
    """Upload the database back, refusing to overwrite a version we did not
    start from. Returns False when S3 is not configured."""
    target = _target()
    if not target:
        return False
    import boto3
    bucket, key = target
    extra = {"IfMatch": _etag} if _etag else {"IfNoneMatch": "*"}
    boto3.client("s3").put_object(Bucket=bucket, Key=key,
                                  Body=Path(local_path).read_bytes(), **extra)
    return True
