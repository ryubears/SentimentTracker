import sys; sys.path.insert(0, "src")
import json, os
from sentiment_tracker import runtime, storage

CFG = 'accounts:\n  - handle: a\nhorizon: "1h"\nsymbol: "BTCUSDT"\ndb_path: "data/t.sqlite"\n'

def test_db_path_resolves_relative_to_the_config_not_the_cwd(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(CFG)
    monkeypatch.delenv("TRACKER_DB_PATH", raising=False)
    monkeypatch.chdir("/")           # a caller elsewhere must not redirect the db
    cfg = runtime.load_config(str(tmp_path / "config.yaml"))
    assert cfg["db_path"] == str(tmp_path / "data" / "t.sqlite")

def test_env_overrides_db_path(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(CFG)
    monkeypatch.setenv("TRACKER_DB_PATH", "/tmp/tracker.sqlite")
    assert runtime.load_config(str(tmp_path / "config.yaml"))["db_path"] == "/tmp/tracker.sqlite"

def test_config_path_can_come_from_the_environment(tmp_path, monkeypatch):
    (tmp_path / "custom.yaml").write_text(CFG)
    monkeypatch.setenv("TRACKER_CONFIG", str(tmp_path / "custom.yaml"))
    monkeypatch.delenv("TRACKER_DB_PATH", raising=False)
    assert runtime.load_config()["horizon"] == "1h"

def test_secrets_manager_never_overrides_an_explicit_env_var(monkeypatch):
    monkeypatch.setenv("TRACKER_SECRET_ID", "some/secret")
    monkeypatch.setenv("X_BEARER_TOKEN", "already-set")
    class FakeSM:
        def get_secret_value(self, SecretId):
            return {"SecretString": json.dumps({"X_BEARER_TOKEN": "from-aws",
                                                "ANTHROPIC_API_KEY": "new-key"})}
    monkeypatch.setitem(sys.modules, "boto3", type("m", (), {"client": staticmethod(lambda n: FakeSM())}))
    loaded = runtime.load_secrets()
    assert os.environ["X_BEARER_TOKEN"] == "already-set"  # explicit export wins
    assert os.environ["ANTHROPIC_API_KEY"] == "new-key" and loaded == ["ANTHROPIC_API_KEY"]

def test_storage_is_a_noop_without_a_bucket(monkeypatch, tmp_path):
    monkeypatch.delenv("TRACKER_S3_BUCKET", raising=False)
    assert storage.pull(str(tmp_path / "x.sqlite")) is False
    assert storage.push(str(tmp_path / "x.sqlite")) is False
