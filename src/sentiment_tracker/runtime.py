"""
Where config, secrets and the database live.

Locally that is a config.yaml beside the code, a .env file, and a SQLite file
under data/. On Lambda the code is read-only at /var/task, the only writable
disk is /tmp, and secrets come from Secrets Manager. Everything that differs
between those two worlds is resolved here so the pipeline itself does not have
to care which one it is running in.

Environment overrides (all optional, all no-ops locally):
  TRACKER_CONFIG     path to config.yaml
  TRACKER_DB_PATH    overrides db_path from the config
  TRACKER_SECRET_ID  Secrets Manager secret whose JSON keys become env vars
"""

from __future__ import annotations
from pathlib import Path
import json
import os
import yaml

# src/sentiment_tracker/runtime.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_secrets() -> list[str]:
    """
    Populate os.environ with API credentials, and return the names loaded.

    Secrets Manager first when TRACKER_SECRET_ID is set (Lambda), then a .env
    file (local). Existing environment variables always win, so an explicitly
    exported value is never silently overridden by either source.
    """
    loaded = []
    secret_id = os.environ.get("TRACKER_SECRET_ID")
    if secret_id:
        import boto3 # only present in the Lambda image; not a local dependency
        raw = boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)["SecretString"]
        for k, v in json.loads(raw).items():
            if k not in os.environ:
                os.environ[k] = str(v)
                loaded.append(k)
        return loaded
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return loaded

def load_config(path: str | None = None) -> dict:
    """Config with environment overrides applied. db_path is resolved to an
    absolute path so a caller's working directory cannot change which file the
    pipeline writes to."""
    path = path or os.environ.get("TRACKER_CONFIG") or str(PROJECT_ROOT / "config.yaml")
    cfg = yaml.safe_load(open(path))
    cfg["db_path"] = os.environ.get("TRACKER_DB_PATH") or cfg["db_path"]
    if not Path(cfg["db_path"]).is_absolute():
        cfg["db_path"] = str((Path(path).resolve().parent / cfg["db_path"]).resolve())
    return cfg
