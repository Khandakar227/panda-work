"""The public repo carries no vocabulary from the study it was extracted from."""
import re
from pathlib import Path

BANNED = ("misfit", "grid2", "pilot", r"\bm7\b", "judge_check")
SKIP_DIRS = {".git", ".venv", "__pycache__", "runs", "outputs", ".pytest_cache", ".superpowers"}
SKIP_PATHS = {Path("docs") / "superpowers"}
PATTERN = re.compile("|".join(BANNED), re.IGNORECASE)


def banned_hits(root: Path) -> list[str]:
    """`path:line: text` for every line under `root` matching a banned word; binary files skipped."""
    hits = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if (not path.is_file() or SKIP_DIRS & set(rel.parts) or any(p.endswith(".egg-info") for p in rel.parts)
                or any(rel.is_relative_to(s) for s in SKIP_PATHS) or path.resolve() == Path(__file__).resolve()):
            continue  # this file names the words on purpose
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:100]}")
    return hits


def test_repo_has_no_study_vocabulary():
    hits = banned_hits(Path(__file__).parents[1])
    assert hits == [], "\n".join(hits)


def test_banned_hits_sees_docstrings_and_test_strings(tmp_path):
    (tmp_path / "x.py").write_text('"""about the MISFIT study"""\nWORD = "pilot"\n')
    (tmp_path / "docs" / "superpowers").mkdir(parents=True)
    (tmp_path / "docs" / "superpowers" / "spec.md").write_text("misfit grid2")
    hits = banned_hits(tmp_path)
    assert len(hits) == 2 and all("x.py" in h for h in hits)
