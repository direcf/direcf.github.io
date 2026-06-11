"""Regenerate all 10 chapter cheatsheets using nanobanana-pro (gemini-3-pro-image-preview).
Better Korean text legibility than the previous gemini-2.5-flash-image attempt."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/Users/1113493/.claude/skills/coursework-creator/scripts")
from generate_cheatsheet import build_prompt  # reuse the exact same prompt shape

ROOT = Path("/Users/1113493/Desktop/Study/Engineering-Philosophy")
OUT_DIR = ROOT / "cheatsheets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NANOBANANA = "/Users/1113493/.claude/skills/nanobanana/scripts/nanobanana.py"

with open(ROOT / "course_data.json", encoding="utf-8") as f:
    data = json.load(f)

for chapter in data["chapters"]:
    n = chapter["number"]
    prompt = build_prompt(chapter)
    output = OUT_DIR / f"chapter-{n:02d}.png"
    print(f"\n=== Chapter {n}: {chapter['title']} ===")
    print(f"  → {output.name}")
    result = subprocess.run(
        [
            "python3", NANOBANANA, "generate",
            "--prompt", prompt,
            "--output", str(output),
            "--model", "nanobanana-pro",
            "--ratio", "9:16",
            "--size", "2K",
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  ✓ saved ({output.stat().st_size // 1024} KB)")
    else:
        print(f"  ✗ failed: {result.stderr[-400:]}")

print("\nDone.")
