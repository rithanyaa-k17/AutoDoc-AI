"""Repeatable evaluation on the SYNTHETIC documents (known questions, expected pages, known changes).

Run from the project root:   python Code/evaluate.py
Outputs are written to Evaluation_Results/ and are REAL measurements of THIS run.
The benchmark is self-authored (synthetic) - it demonstrates correct behaviour on controlled data
and does NOT prove accuracy on real HLD documents.
"""
import csv
import datetime
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pymupdf
import sklearn

from autodoc.checks import run_checks
from autodoc.compare import compare_docs
from autodoc.config import INPUT_DIR, RESULTS_DIR
from autodoc.extract import build_inventory
from autodoc.ingest import ingest_pdf
from autodoc.retrieval import Index, answer

V10, V11 = INPUT_DIR / "SYNTHETIC_Wiper_HLD_v1.0.pdf", INPUT_DIR / "SYNTHETIC_Wiper_HLD_v1.1.pdf"
GT_PATH = INPUT_DIR / "synthetic_ground_truth.json"
REAL = INPUT_DIR / "AUTOSAR_EXP_SWArchitecture.pdf"


def write_csv(name, rows):
    if not rows:
        return
    with open(RESULTS_DIR / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    if not (V10.exists() and V11.exists() and GT_PATH.exists()):
        print("Synthetic files missing. Run first:  python Code/generate_synthetic.py")
        return 1
    RESULTS_DIR.mkdir(exist_ok=True)
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    d10 = ingest_pdf(V10.name, V10.read_bytes())
    d11 = ingest_pdf(V11.name, V11.read_bytes())
    results = {"run_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "python": platform.python_version(), "pymupdf": pymupdf.__version__,
               "scikit_learn": sklearn.__version__, "benchmark": "synthetic, self-authored",
               "ingestion": {d["name"]: {"status": d["status"], "pages": d["page_count"]} for d in (d10, d11)}}

    # ---- 1. Q&A
    idx = Index([d10])
    qa_rows, passed = [], 0
    for item in gt["qa"]:
        res = answer(idx, item["q"])
        top3 = res["evidence"][:3]
        page_ok = any(h["page"] in item["pages"] for h in top3)
        phrase_ok = any(item["phrase"].lower() in h["text"].lower() for h in top3)
        ok = res["status"] == "ok" and page_ok and phrase_ok
        passed += ok
        qa_rows.append({"type": "positive", "question": item["q"], "expected_pages": item["pages"],
                        "status": res["status"], "top_score": round(res["top_score"], 3),
                        "retrieved_pages_top3": [h["page"] for h in top3], "expected_page_in_top3": page_ok,
                        "expected_phrase_in_top3": phrase_ok, "PASS": ok})
    neg_pass = 0
    for q in gt["negative_questions"]:
        res = answer(idx, q)
        ok = res["status"] == "insufficient"
        neg_pass += ok
        qa_rows.append({"type": "negative", "question": q, "expected_pages": "-", "status": res["status"],
                        "top_score": round(res["top_score"], 3), "retrieved_pages_top3": [],
                        "expected_page_in_top3": "-", "expected_phrase_in_top3": "-", "PASS": ok})
    results["qa"] = {"positive_passed": f"{passed}/{len(gt['qa'])}",
                     "negative_correctly_refused": f"{neg_pass}/{len(gt['negative_questions'])}"}
    write_csv("qa_results.csv", qa_rows)

    # ---- 2. Extraction vs ground truth
    inv, per_sentence = build_inventory([d10])
    ext_rows, ext = [], {}
    for etype, key in (("component", "components"), ("interface", "interfaces"),
                       ("port", "ports"), ("signal", "signals")):
        found = {r["entity"] for r in inv if r["type"] == etype}
        truth = set(gt[key])
        tp, fp, fn = found & truth, found - truth, truth - found
        prec = len(tp) / len(found) if found else 0.0
        rec = len(tp) / len(truth) if truth else 0.0
        ext[etype] = {"found": len(found), "truth": len(truth), "true_pos": len(tp),
                      "false_pos": sorted(fp), "missed": sorted(fn),
                      "precision": round(prec, 2), "recall": round(rec, 2)}
        ext_rows.append({"entity_type": etype, "found": len(found), "ground_truth": len(truth),
                         "true_positives": len(tp), "false_positives": "; ".join(sorted(fp)),
                         "missed": "; ".join(sorted(fn)), "precision": round(prec, 2), "recall": round(rec, 2)})
    results["extraction"] = ext
    write_csv("extraction_results.csv", ext_rows)

    # ---- 3. Change detection
    rows, summ, impact = compare_docs(d10, d11)
    cmp_rows, found_n = [], 0
    for exp in gt["expected_changes_v1_0_to_v1_1"]:
        hit = any(r["change_type"] == exp["type"] and
                  exp["contains"].lower() in (r["before_text"] + " " + r["after_text"]).lower() for r in rows)
        found_n += hit
        cmp_rows.append({"expected_type": exp["type"], "expected_text_contains": exp["contains"], "DETECTED": hit})
    impact_names = {i["element"] for i in impact}
    miss_el = [e for e in gt["expected_affected_elements"] if e not in impact_names]
    results["comparison"] = {"expected_changes_detected": f"{found_n}/{len(gt['expected_changes_v1_0_to_v1_1'])}",
                             "total_changes_reported": len(rows), "summary": summ,
                             "expected_elements_flagged": f"{len(gt['expected_affected_elements']) - len(miss_el)}/"
                                                          f"{len(gt['expected_affected_elements'])}",
                             "note": "Reported changes beyond the expected list are real edits too (e.g. version line)."}
    write_csv("comparison_expected.csv", cmp_rows)
    write_csv("comparison_all_changes.csv", rows)

    # ---- 4. Inconsistency checks
    findings = run_checks(per_sentence)
    chk_rows, chk_n = [], 0
    for rule, needle in gt["expected_inconsistency_rules_v1_0"].items():
        hit = any(f["rule"] == rule and needle in f["finding"] for f in findings)
        chk_n += hit
        chk_rows.append({"rule": rule, "expected_to_mention": needle, "DETECTED": hit})
    results["inconsistency_checks"] = {"expected_detected": f"{chk_n}/{len(gt['expected_inconsistency_rules_v1_0'])}",
                                       "total_findings": len(findings),
                                       "all_findings": [f"{f['rule']}: {f['finding']}" for f in findings]}
    write_csv("checks_expected.csv", chk_rows)
    write_csv("checks_all_findings.csv", findings)

    # ---- 5. Real document smoke run (NO accuracy claim; for manual verification)
    if REAL.exists():
        t0 = time.time()
        dr = ingest_pdf(REAL.name, REAL.read_bytes())
        ir = Index([dr])
        rinv, _ = build_inventory([dr])
        manual = []
        for q in ["What is the Execution Management responsible for?", "How does Communication Management work?",
                  "What is the role of Diagnostic Management?", "What is a Functional Cluster?",
                  "What is the recipe for chocolate cake?"]:
            res = answer(ir, q)
            top = res["evidence"][0] if res["evidence"] else None
            manual.append({"question": q, "status": res["status"], "confidence": res["confidence"],
                           "top_page": top["page"] if top else "", "top_excerpt": top["text"][:400] if top else "",
                           "STUDENT_VERDICT (correct / partly / wrong)": ""})
        write_csv("real_document_manual_review.csv", manual)
        results["real_document_smoke_run"] = {"status": dr["status"], "pages": dr["page_count"],
                                              "chunks": len(ir.chunks), "candidate_entities": len(rinv),
                                              "seconds": round(time.time() - t0, 1),
                                              "note": "No accuracy claimed. Fill STUDENT_VERDICT in real_document_manual_review.csv."}
    else:
        results["real_document_smoke_run"] = "skipped (AUTOSAR_EXP_SWArchitecture.pdf not found in Input_Data)"

    (RESULTS_DIR / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = ["# Evaluation summary (generated by Code/evaluate.py)", "",
          f"Run at: {results['run_at']} | Python {results['python']} | PyMuPDF {results['pymupdf']} | "
          f"scikit-learn {results['scikit_learn']}", "",
          "Benchmark: synthetic, self-authored documents. Results show behaviour on controlled data only.", "",
          f"- Q&A positive questions passed: **{results['qa']['positive_passed']}**",
          f"- Out-of-scope questions correctly refused: **{results['qa']['negative_correctly_refused']}**",
          f"- Expected document changes detected: **{results['comparison']['expected_changes_detected']}** "
          f"(total reported: {len(rows)}; {summ})",
          f"- Expected inconsistency findings detected: **{results['inconsistency_checks']['expected_detected']}** "
          f"(total findings: {len(findings)})", "", "## Extraction precision / recall vs ground truth", ""]
    for k, v in ext.items():
        md.append(f"- {k}: precision {v['precision']}, recall {v['recall']} (false positives: {v['false_pos'] or 'none'}; "
                  f"missed: {v['missed'] or 'none'})")
    (RESULTS_DIR / "eval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("QA positive:", results["qa"]["positive_passed"], "| negative refused:", results["qa"]["negative_correctly_refused"])
    print("Changes detected:", results["comparison"]["expected_changes_detected"], "| elements flagged:",
          results["comparison"]["expected_elements_flagged"])
    print("Inconsistencies detected:", results["inconsistency_checks"]["expected_detected"])
    for k, v in ext.items():
        print(f"Extraction {k}: P={v['precision']} R={v['recall']} FP={v['false_pos']} missed={v['missed']}")
    print("Saved to", RESULTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
