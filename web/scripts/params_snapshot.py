"""Write main's /params/defaults answer into the browser build (browser/params.defaults.json)."""

import json
import sys
from pathlib import Path

from minime.serve.paramsmeta import GROUPS, defaults

Path(sys.argv[1]).write_text(
    json.dumps({"groups": list(GROUPS), "params": defaults()}, indent=1) + "\n"
)
