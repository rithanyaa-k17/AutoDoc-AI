"""Setup / health check.   python Code\\check_setup.py            (quick check)
                           python Code\\check_setup.py --download (also downloads + tests the embedding model)
Prints what works and exactly what to do about what does not. Never prints API keys."""
import argparse
import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def ver(mod):
    try:
        m = importlib.import_module(mod)
        return True, getattr(m, "__version__", getattr(m, "VersionBind", "ok"))
    except Exception as exc:
        return False, str(exc)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="load (and download once) the embedding model")
    a = ap.parse_args()
    print(f"Python {platform.python_version()} on {platform.system()}")
    problems = []
    for label, mod, required in [("PyMuPDF", "pymupdf", True), ("scikit-learn", "sklearn", True),
                                 ("pandas", "pandas", True), ("numpy", "numpy", True),
                                 ("streamlit", "streamlit", True),
                                 ("sentence-transformers", "sentence_transformers", True),
                                 ("faiss-cpu (optional)", "faiss", False)]:
        ok, v = ver(mod)
        print(f"  [{'OK' if ok else ('MISSING' if required else 'not installed')}] {label}: {v if ok else ''}")
        if not ok and required:
            problems.append(f"python -m pip install -r requirements.txt   (missing {label})")
    from autodoc.config import CFG
    from autodoc.llm import LLMClient
    print(f"Embedding model (configured): {CFG['rag']['embedding_model']}")
    if a.download:
        try:
            from autodoc.embeddings import SentenceTransformerEmbedder
            e = SentenceTransformerEmbedder()
            v = e.encode(["wiper control software component"])
            print(f"  [OK] embedding model loaded, vector dimension {v.shape[1]}")
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            problems.append("Embedding model not available (needs internet ONCE to download ~90 MB).")
    else:
        print("  (not loaded in quick mode; use --download once while online)")
    llm = LLMClient()
    h = llm.health()
    print(f"LLM provider: {llm.label()}")
    print(f"  [{'OK' if h['ok'] else 'NOT READY'}] {h['detail']}")
    if not h["ok"]:
        problems.append("LLM not ready: see README section 'Choose your LLM' (Ollama or hosted endpoint). "
                        "The app still runs in a clearly labelled retrieval-only mode.")
    print()
    print("ALL GOOD - run: streamlit run Code\\app.py" if not problems else "TO FIX:")
    for p in problems:
        print("  -", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
