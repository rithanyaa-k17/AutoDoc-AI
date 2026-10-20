"""Create the submission ZIP.  Run from the project root:
    python Code\\package_zip.py --name UniversityName_StudentName_RegisterNo_CSx_AIML
The ZIP is written NEXT TO the project folder (so it never contains itself); .venv and caches are excluded.
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", ".pytest_cache", ".streamlit_cache", ".cache_index", "hf_cache", "models"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="UniversityName_StudentName_RegisterNo_CSx_AIML")
    name = ap.parse_args().name
    out = ROOT.parent / f"{name}.zip"
    warn = []
    for f in (ROOT / "Declarations").glob("*.md"):
        t = f.read_text(encoding="utf-8", errors="ignore")
        if "[YOUR NAME]" in t or "[SIGNATURE]" in t:
            warn.append(f.name)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if p.is_dir() or any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
                continue
            z.write(p, Path(name) / p.relative_to(ROOT))
            n += 1
    print(f"Created {out} ({n} files)")
    for w in warn:
        print(f"WARNING: Declarations/{w} still contains placeholders - complete and sign it before submitting.")
    ev = ROOT / "Evaluation_Results" / "eval_results.json"
    if ev.exists() and "SANDBOX" in ev.read_text(encoding="utf-8", errors="ignore"):
        print("WARNING: Evaluation_Results came from a SANDBOX stand-in, not real PyMuPDF. Re-run "
              "evaluate.py and generate_report.py on your machine before submitting.")
    for need in ("Video", "Evaluation_Results", "Documentation/Technical_Report.pdf"):
        p = ROOT / need
        if (p.is_dir() and not any(x.name != "README.md" for x in p.iterdir())) or (not p.exists()):
            print(f"WARNING: {need} looks empty/missing - expected before submission.")


if __name__ == "__main__":
    main()
