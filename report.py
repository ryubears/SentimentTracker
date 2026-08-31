"""
Print evaluation metrics and refresh RESULTS.md.
"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from sentiment_tracker import db, evaluate, runtime

cfg = runtime.load_config()
res = evaluate.evaluate(db.connect(cfg["db_path"]),
                        deadband=cfg.get("signal", {}).get("deadband", 0.0))
print(json.dumps(res, indent=2))
open("RESULTS.md", "w").write("# Live results\n\n```json\n" + json.dumps(res, indent=2) + "\n```\n")
