"""
Print evaluation metrics; used by the GitHub Action to refresh RESULTS.md.
"""

import json, sys
sys.path.insert(0, "src")
import yaml
from sentiment_tracker import db, evaluate

cfg = yaml.safe_load(open("config.yaml"))
res = evaluate.evaluate(db.connect(cfg["db_path"]))
print(json.dumps(res, indent=2))
open("RESULTS.md", "w").write("# Live results\n\n```json\n" + json.dumps(res, indent=2) + "\n```\n")
