"""LLM + RAG integration evaluation. REQUIRES a real embedding model and a reachable LLM provider.
    python Code\\evaluate_rag.py
Runs the synthetic questions through the full pipeline and writes Evaluation_Results\\rag_llm_results.csv and
rag_llm_summary.json. If prerequisites are missing it exits WITHOUT writing results (nothing is invented).
Synthetic questions only: this is a smoke/regression check of grounding behaviour, NOT real-world accuracy."""
import csv
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autodoc.config import CFG, INPUT_DIR, RESULTS_DIR
from autodoc.ingest import ingest_pdf
from autodoc.rag import RagEngine


def run(embedder, llm, out_dir: Path, min_score=None):
    pdf = INPUT_DIR / "SYNTHETIC_Wiper_HLD_v1.0.pdf"
    gt = json.loads((INPUT_DIR / "synthetic_ground_truth.json").read_text(encoding="utf-8"))
    doc = ingest_pdf(pdf.name, pdf.read_bytes())
    eng = RagEngine([doc], embedder=embedder, llm=llm, use_disk_cache=False)
    rows = []
    for item in gt["qa"]:
        r = eng.ask(item["q"], min_score=min_score)
        cited_pages = sorted({c["page"] for c in r["citations"]})
        retrieved_pages = sorted({h["page"] for h in r["evidence"]})
        phrase_in_evidence = any(item["phrase"].lower() in h["text"].lower() for h in r["evidence"])
        rows.append({"type": "positive", "question": item["q"], "mode": r["mode"], "status": r["status"],
                     "answer": r["answer"].replace("\n", " "), "cited_pages": cited_pages,
                     "retrieved_pages": retrieved_pages, "expected_pages": item["pages"],
                     "has_valid_citation": bool(r["citations"]),
                     "expected_page_retrieved": bool(set(item["pages"]) & set(retrieved_pages)),
                     "expected_phrase_in_evidence": phrase_in_evidence,
                     "warnings": "; ".join(r["warnings"]),
                     "STUDENT_VERDICT (correct/partly/wrong)": ""})
    for q in gt["negative_questions"]:
        r = eng.ask(q, min_score=min_score)
        rows.append({"type": "negative", "question": q, "mode": r["mode"], "status": r["status"],
                     "answer": (r["answer"] or r["message"]).replace("\n", " "), "cited_pages": [],
                     "retrieved_pages": sorted({h["page"] for h in r["evidence"]}), "expected_pages": "-",
                     "has_valid_citation": "-", "expected_page_retrieved": "-", "expected_phrase_in_evidence": "-",
                     "warnings": "; ".join(r["warnings"]), "STUDENT_VERDICT (correct/partly/wrong)": ""})
    pos = [r for r in rows if r["type"] == "positive"]
    neg = [r for r in rows if r["type"] == "negative"]
    summary = {"run_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "embedding_model": embedder.name, "llm": llm.label(), "vector_backend": eng.store.backend,
               "benchmark": "synthetic, self-authored (smoke test; not real-world accuracy)",
               "positive_questions": len(pos),
               "answers_generated_by_llm": sum(r["mode"] == "llm_rag" and r["status"] == "ok" for r in pos),
               "answers_with_valid_citation": sum(r["has_valid_citation"] is True for r in pos),
               "expected_page_retrieved": sum(r["expected_page_retrieved"] is True for r in pos),
               "negative_questions": len(neg),
               "negatives_refused_or_insufficient": sum(r["status"] == "insufficient" for r in neg),
               "dense_min_score": CFG["rag"]["dense_min_score"] if min_score is None else min_score}
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "rag_llm_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "rag_llm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, rows


def main():
    from autodoc.embeddings import EmbeddingUnavailable, SentenceTransformerEmbedder
    from autodoc.llm import LLMClient
    if not (INPUT_DIR / "SYNTHETIC_Wiper_HLD_v1.0.pdf").exists():
        print("Run first: python Code\\generate_synthetic.py")
        return 2
    try:
        emb = SentenceTransformerEmbedder()
    except EmbeddingUnavailable as exc:
        print("Cannot run: embedding model unavailable.\n", exc)
        return 2
    llm = LLMClient()
    h = llm.health()
    if not h["ok"]:
        print("Cannot run: LLM not ready.\n", h["detail"])
        return 2
    summary, rows = run(emb, llm, RESULTS_DIR)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("Saved rag_llm_results.csv (fill STUDENT_VERDICT by reading each answer) and rag_llm_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
