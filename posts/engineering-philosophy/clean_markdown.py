"""Strip `**bold**` and other stray markdown markers from course_data.json
so the HTML builder (which escapes everything) produces clean prose."""
import json
import re
from pathlib import Path

PATH = Path("/Users/1113493/Desktop/Study/Engineering-Philosophy/course_data.json")

# Preserve code blocks exactly — never touch `codeExample.code`.
CODE_PATH_PARTS = {"code"}

BOLD = re.compile(r"\*\*(.+?)\*\*")          # **x** → x
ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])")  # *x* → x  (conservative)

def clean(s: str) -> str:
    s = BOLD.sub(r"\1", s)
    s = ITALIC.sub(r"\1", s)
    return s

def walk(obj, key=None):
    if isinstance(obj, dict):
        return {k: walk(v, key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk(v, key=key) for v in obj]
    if isinstance(obj, str) and key not in CODE_PATH_PARTS:
        return clean(obj)
    return obj

data = json.loads(PATH.read_text(encoding="utf-8"))
cleaned = walk(data)

# Backup once, then overwrite.
backup = PATH.with_suffix(".json.bak")
if not backup.exists():
    backup.write_text(PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"backup → {backup.name}")

PATH.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"cleaned → {PATH.name}")
