"""Generate Documentation/Technical_Report.pdf (+ .md copy) and workflow/architecture diagrams.

Run from the project root:   python Code/generate_report.py
Evaluation numbers are read from Evaluation_Results/eval_results.json. If that file does not exist the
report says PENDING - run  python Code/evaluate.py  first, then regenerate the report.
Fill the placeholders in [SQUARE BRACKETS] in PLACEHOLDERS below (or edit the PDF source) before submitting.
"""
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "Documentation"
DIAG = DOC / "diagrams"
EVAL = ROOT / "Evaluation_Results" / "eval_results.json"
RAGEVAL = ROOT / "Evaluation_Results" / "rag_llm_summary.json"

PLACEHOLDERS = {"name": "[YOUR NAME]", "reg": "[REGISTER NO]", "uni": "[UNIVERSITY NAME]",
                "cs": "[CSx]", "date": "[SUBMISSION DATE]"}


# ------------------------------------------------------------------ diagrams
def _box(ax, x, y, w, h, text, fc="#e8f0fe", fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#33475b", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#33475b", lw=1.2))


def make_diagrams():
    DIAG.mkdir(parents=True, exist_ok=True)
    # ---- workflow (two rows)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 4.6)
    ax.axis("off")
    top = ["Upload PDF(s)", "PyMuPDF text\nper page\n(+page numbers)", "Overlapping chunks\n<=90 words, 1 sentence\noverlap, file+page+\nsection metadata",
           "Sentence-Transformers\nall-MiniLM-L6-v2\nembeddings", "Vector store\nFAISS or NumPy\n(cached on disk)"]
    for i, t in enumerate(top):
        x = 0.1 + i * 2.0
        _box(ax, x, 3.0, 1.7, 1.3, t, "#e8f0fe")
        if i:
            _arrow(ax, x - 0.3, 3.65, x, 3.65)
    bot = ["Question ->\nquery embedding\n-> top-k cosine", "Evidence gate\n(min similarity)\nbelow -> REFUSE,\nLLM not called", "Controlled prompt\n(context-only rules,\n[S#] citations)",
           "LLM (Ollama /\nOpenAI-compatible)", "Citation check +\nanswer mode label\n+ evidence panel"]
    cols = ["#e8f0fe", "#fff4e5", "#fff4e5", "#e6f4ea", "#e6f4ea"]
    for i, t in enumerate(bot):
        x = 0.1 + i * 2.0
        _box(ax, x, 1.1, 1.7, 1.3, t, cols[i])
        if i:
            _arrow(ax, x - 0.3, 1.75, x, 1.75)
    _arrow(ax, 8.95, 3.0, 1.0, 2.4)
    _box(ax, 0.1, -0.3, 9.6, 0.7, "No embeddings or no LLM available -> labelled fallback (TF-IDF extractive / retrieval-only), never shown as LLM output."
         "   Human review + CSV/JSON export.", "#fce8e6", fs=7)
    ax.set_title("AutoDoc AI RAG workflow", fontsize=10)
    fig.savefig(DIAG / "workflow.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    # ---- architecture
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    _box(ax, 2, 7.1, 6, 0.7, "Engineer (browser) - Streamlit UI (status panel + 7 tabs)", "#e6f4ea", 8)
    _box(ax, 2, 6.0, 6, 0.8, "Code/app.py  (cached embedder, cached index, session state)", "#e8f0fe", 8)
    _arrow(ax, 5, 7.1, 5, 6.8)
    mods = [("ingest.py\nPyMuPDF", 0.2), ("rag.py\nprompt + RAG\nengine", 2.2), ("embeddings.py\nMiniLM", 4.2),
            ("vectorstore.py\nFAISS/NumPy", 6.2), ("llm.py\nOpenAI-compat.\nclient", 8.2)]
    for t, x in mods:
        _box(ax, x - 0.1 if x == 0.2 else x - 0.1, 4.2, 1.7, 1.2, t, "#fff4e5")
        _arrow(ax, 5, 6.0, x + 0.75, 5.4)
    _box(ax, 0.1, 2.5, 3.0, 1.1, "Legacy/analysis modules:\nretrieval.py (TF-IDF fallback),\nextract.py, compare.py, checks.py", "#f3e8fd")
    _box(ax, 3.4, 2.5, 3.0, 1.1, "Model_Prompts_Config/\nconfig.json + system_prompt.txt\n(no secrets)", "#f3e8fd")
    _box(ax, 6.7, 2.5, 3.2, 1.1, "Env vars: AUTODOC_LLM_*\n(API key never written to disk)", "#f3e8fd")
    _box(ax, 0.1, 0.9, 4.7, 1.0, "Local files: Input_Data/ PDFs,\n.cache_index/ (vectors + chunk metadata)", "#f3e8fd")
    _box(ax, 5.1, 0.9, 4.8, 1.0, "LLM server: Ollama on localhost (local)\nor any OpenAI-compatible endpoint (optional)", "#e6f4ea")
    ax.set_title("AutoDoc AI component architecture", fontsize=10)
    fig.savefig(DIAG / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ content
def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def eval_blocks():
    B = []
    r = _load(EVAL)
    B.append(("h2", "5.1 Deterministic offline tests and synthetic evaluation (TF-IDF analysis modules)"))
    if not r:
        B.append(("p", "<b>PENDING.</b> Run <font face='Courier'>python Code\\evaluate.py</font> then regenerate this report. No result is claimed."))
    else:
        B.append(("p", f"Measured {escape(r['run_at'])} (Python {r['python']}, PyMuPDF {r['pymupdf']}, scikit-learn {r['scikit_learn']}) on <b>synthetic, "
                       "self-authored</b> documents. Controlled-data behaviour only; not real-world accuracy."))
        rows = [["Test", "Measured result"],
                ["Legacy TF-IDF Q&A: expected page in top-3 + phrase", r["qa"]["positive_passed"]],
                ["Out-of-scope questions refused", r["qa"]["negative_correctly_refused"]],
                ["Planted document changes detected", r["comparison"]["expected_changes_detected"]],
                ["Planted inconsistencies detected", r["inconsistency_checks"]["expected_detected"]]]
        for k, v in r["extraction"].items():
            rows.append([f"Extraction ({k}) precision / recall", f"{v['precision']} / {v['recall']}"])
        B.append(("table", rows))
    B.append(("p", "Offline unit tests (<font face='Courier'>python -m unittest discover -s Code/tests -v</font>) cover chunking, page metadata, "
                   "vector-store round trip, FAISS/NumPy agreement (if faiss is installed), prompt construction, citation checking, grounded answers "
                   "with a <b>mock</b> OpenAI-compatible server, refusal without calling the LLM, fallback labelling, and invalid/encrypted/blank PDFs. "
                   "They use a hashed bag-of-words test double instead of the real embedding model, so they verify the pipeline plumbing and guardrails, "
                   "<b>not</b> semantic retrieval quality or LLM answer quality."))
    B.append(("h2", "5.2 LLM + RAG integration run (real embedding model + real LLM)"))
    g = _load(RAGEVAL)
    if not g:
        B.append(("p", "<b>PENDING.</b> Requires the real embedding model and a reachable LLM. Run <font face='Courier'>python Code\\evaluate_rag.py</font> and regenerate this report."))
    else:
        B.append(("table", [["Item", "Measured (this run)"], ["Run at", g["run_at"]], ["Embedding model", g["embedding_model"]],
                            ["LLM", g["llm"]], ["Vector backend", g["vector_backend"]],
                            ["Positive questions answered by LLM", f"{g['answers_generated_by_llm']}/{g['positive_questions']}"],
                            ["... with a valid [S#] citation", f"{g['answers_with_valid_citation']}/{g['positive_questions']}"],
                            ["... expected page among retrieved", f"{g['expected_page_retrieved']}/{g['positive_questions']}"],
                            ["Out-of-scope questions refused", f"{g['negatives_refused_or_insufficient']}/{g['negative_questions']}"],
                            ["Minimum-similarity threshold", g["dense_min_score"]]]))
        B.append(("p", "Synthetic questions only: a smoke test of grounding behaviour. Answer correctness was not auto-graded; "
                       "see the STUDENT_VERDICT column in rag_llm_results.csv."))
    B.append(("h2", "5.3 Real-document manual validation"))
    B.append(("p", "The official AUTOSAR reference PDF (Adaptive Platform explanatory document) is <b>not</b> a project-specific HLD; it is used only to check that "
                   "ingestion, retrieval and citations behave on a long real document. Verdicts are recorded manually by the student in "
                   "real_document_manual_review.csv; no accuracy figure is claimed."))
    return B


def blocks():
    P = PLACEHOLDERS
    B = [("title", "AutoDoc AI - AUTOSAR HLD Document Analysis Assistant (local RAG)"),
         ("p", f"Technical Report | Case Study 1 (TechPulse / Tata Technologies internship)<br/>{P['name']} | {P['uni']} | Register No. {P['reg']} | {P['cs']} | {P['date']}"),
         ("h1", "1. Problem statement and intended users"),
         ("p", "AUTOSAR High-Level Design (HLD) documents are long unstructured PDFs. Architects, developers, integrators and testers read them manually to find "
               "components, interfaces, signals, flows and version changes (Case Study 1). AutoDoc AI is a prototype that lets an engineer ask questions in natural language and "
               "receive answers grounded in the uploaded document with page citations, extracts <i>candidate</i> architecture elements, compares versions and reports "
               "<i>potential</i> inconsistencies. Intended users: system/AUTOSAR architects, developers, integration and test engineers, engineering managers. Out of scope "
               "(per case study): automatic approval of architecture decisions, modifying source documents, replacing architecture governance."),
         ("h1", "2. Input data and knowledge base"),
         ("p", "Knowledge base = the PDFs the user loads. Inputs used: (a) <b>SYNTHETIC_Wiper_HLD_v1.0.pdf / v1.1.pdf</b> - fictional HLDs generated by Code/generate_synthetic.py "
               "(planted inconsistencies and changes; not official AUTOSAR or Tata Technologies material) - the primary demo and evaluation data; (b) the public AUTOSAR "
               "<i>Explanation of Adaptive Platform Software Architecture</i> (R22-11) as a long real-document robustness test. <b>The AUTOSAR reference PDF is not a project-specific HLD</b> "
               "and is not presented as one. Ingestion: PyMuPDF text per page, running header/footer removal, numbered-heading detection, table-of-contents exclusion; scanned pages are "
               "detected and flagged (OCR optional, not required)."),
         ("h1", "3. Solution architecture"),
         ("img", "architecture.png"),
         ("table", [["Layer", "Implementation", "Status"],
                    ["Chunking", "Sentence-aligned chunks, max 90 words, 1-sentence overlap, never across page/section; metadata: file, page, section", "Implemented"],
                    ["Embeddings", "sentence-transformers, default all-MiniLM-L6-v2 (384-d, CPU), L2-normalised; downloaded once, cached by Hugging Face cache", "Implemented (needs one-time download)"],
                    ["Vector store", "FAISS IndexFlatIP if faiss-cpu installed, else exact NumPy cosine search; vectors + chunk metadata cached on disk (.cache_index)", "Implemented (FAISS optional)"],
                    ["Retrieval", "Query embedded with the same model; top-k cosine; evidence gate (minimum similarity, adjustable)", "Implemented"],
                    ["LLM", "Any OpenAI-compatible /chat/completions endpoint via urllib: Ollama (local, default llama3.2:3b) or hosted endpoint via env vars", "Implemented; needs Ollama or endpoint"],
                    ["Prompt grounding", "Context-only rules, untrusted-context rule, mandatory [S#] citations, exact INSUFFICIENT_EVIDENCE reply, temperature 0", "Implemented"],
                    ["Answer validation", "Cited tags checked against retrieved passages; uncited/invalid-citation answers flagged", "Implemented"],
                    ["Fallbacks", "No LLM -> retrieval-only excerpts; no embeddings -> TF-IDF extractive; each labelled, never shown as LLM output", "Implemented"],
                    ["In-process local GGUF (llama-cpp-python)", "Not implemented (Windows build/wheel risk on Python 3.13); Ollama provides local llama.cpp inference instead", "Not implemented"],
                    ["FastAPI, SQLite/PostgreSQL, graph store, RBAC, audit log", "Not implemented", "Planned"]]),
         ("h1", "4. Implementation and configuration"),
         ("img", "workflow.png"),
         ("p", "<b>Answer modes</b> (always displayed): <i>LLM + RAG</i> (genuine); <i>REFUSED</i> (similarity below threshold, LLM not called); <i>RETRIEVAL ONLY</i> (LLM absent or failed - excerpts, "
               "nothing generated); <i>FALLBACK TF-IDF</i> (embeddings unavailable). A status panel shows the provider/model, reachability, embedding model, index size, backend and cache state, "
               "and whether genuine LLM+RAG is active."),
         ("p", "<b>Configuration:</b> Model_Prompts_Config/config.json (chunk size/overlap, top-k, minimum similarity, model names, timeouts) and environment variables "
               "AUTODOC_LLM_PROVIDER / _BASE_URL / _MODEL / _API_KEY, AUTODOC_EMBED_MODEL. API keys are never stored in files. The exact system prompt is exported to "
               "Model_Prompts_Config/system_prompt.txt. The default minimum similarity (0.22) is an <b>untuned starting value</b> for MiniLM; adjust it in the sidebar."),
         ("p", "<b>Retained analysis features</b> (rule-based, not LLM): candidate component/interface/port/signal extraction, relationship candidates, functional-flow evidence, "
               "Change Impact Radar (difflib), four inconsistency rules, CSV/JSON export with a human review column."),
         ("h1", "5. Evaluation steps and results")]
    B += eval_blocks()
    B += [("h1", "6. Responsible AI measures"),
          ("table", [["Principle", "Measure"],
                     ["Grounding", "Context-only prompt, mandatory page-tagged citations, citation validation, evidence always shown, refusal when similarity is low (LLM not called)"],
                     ["Honesty about modes", "Answer mode shown on every answer; fallbacks never labelled as LLM"],
                     ["Privacy", "Documents and embeddings stay local. With Ollama nothing leaves the machine. With a hosted endpoint the retrieved passages ARE sent to that provider - use only synthetic/public documents"],
                     ["Security", "PDF-only upload; encrypted/invalid files rejected safely; retrieved text treated as untrusted (prompt-injection rule - a mitigation, not a guarantee); no keys in code or config"],
                     ["Human oversight", "Review-status column, candidate labels, no automatic approvals; LLM output must be verified against the evidence"],
                     ["Limits stated", "LLM can still hallucinate or mis-cite; extraction and checks are heuristic; not AUTOSAR compliance verification"]]),
          ("h1", "7. Innovation highlights"),
          ("p", "Modest, honestly scoped: (1) a refusal gate that avoids calling the LLM on irrelevant questions; (2) post-hoc citation validation of LLM answers; (3) explicit answer-mode labelling with graceful degradation; "
                "(4) provider-agnostic LLM client with zero extra dependencies; (5) Change Impact Radar linking text changes to extracted elements; (6) transparent rule-based inconsistency checks; "
                "(7) a synthetic benchmark with planted ground truth and a mock-LLM offline test suite."),
          ("h1", "8. Limitations and future work"),
          ("p", "Not verified by the author at build time: real embedding-model download/behaviour, Ollama or hosted-LLM answer quality, Windows 11 / Python 3.13 installation of PyTorch and sentence-transformers. "
                "The similarity threshold is untuned. Tables and diagrams are not interpreted. Answer quality depends on the chosen small model. Sentence-based chunking may split table rows. "
                "Future: tuned thresholds, hybrid dense+TF-IDF retrieval, cross-encoder re-ranking, table extraction, in-process GGUF model, FastAPI service layer, SQLite review log, RBAC and audit logs, "
                "graph traceability, larger evaluation on authorised real HLDs."),
          ("h1", "9. References, licences and declarations"),
          ("p", "AUTOSAR 'Explanation of Adaptive Platform Software Architecture' R22-11 (input document, AUTOSAR copyright; not redistributed here). Case-study document 'Automotive Engineering AI - Project Case Studies' v1.0. "
                "Libraries (see requirements.txt): PyMuPDF (AGPL-3.0 / commercial), scikit-learn (BSD), pandas (BSD), NumPy (BSD), Streamlit (Apache-2.0), sentence-transformers (Apache-2.0), PyTorch (BSD-style), "
                "faiss-cpu (MIT, optional), ReportLab (BSD), Matplotlib (PSF-style). Model: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0). LLMs run via Ollama (MIT) with models under their own licences "
                "(e.g. Llama 3.2 Community License) - check the licence of the model you pull. Signed declarations: Declarations/ folder; AI tools used (including coding assistance) are declared there.")]
    return B


# ------------------------------------------------------------------ renderers
def to_pdf(B, path):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10)
    story = []
    for kind, val in B:
        if kind == "title":
            story += [Paragraph(escape(val), ss["Title"]), Spacer(1, 6)]
        elif kind == "h1":
            story += [Spacer(1, 8), Paragraph(escape(val), ss["Heading2"])]
        elif kind == "h2":
            story += [Spacer(1, 4), Paragraph("<b>" + escape(val) + "</b>", body)]
        elif kind == "p":
            story += [Paragraph(val, body), Spacer(1, 5)]
        elif kind == "img":
            img = Image(str(DIAG / val))
            img._restrictSize(17 * cm, 11 * cm)
            story += [img, Spacer(1, 6)]
        elif kind == "table":
            data = [[Paragraph(escape(str(c)), cell) for c in row] for row in val]
            t = Table(data, repeatRows=1, colWidths=None)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [t, Spacer(1, 6)]

    def footer(c, d):
        c.setFont("Helvetica", 8)
        c.drawString(2 * cm, 1 * cm, f"AutoDoc AI technical report | page {d.page}")

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm, title="AutoDoc AI Technical Report").build(
        story, onFirstPage=footer, onLaterPages=footer)


def to_md(B, path):
    import re
    strip = lambda s: re.sub(r"<[^>]+>", "", s).replace("&amp;", "&")
    out = []
    for kind, val in B:
        if kind == "title":
            out.append(f"# {val}\n")
        elif kind == "h1":
            out.append(f"\n## {val}\n")
        elif kind == "h2":
            out.append(f"\n### {val}\n")
        elif kind == "p":
            out.append(strip(val.replace("<br/>", "  \n")) + "\n")
        elif kind == "img":
            out.append(f"![diagram](diagrams/{val})\n")
        elif kind == "table":
            out.append("| " + " | ".join(val[0]) + " |\n|" + "---|" * len(val[0]))
            out += ["| " + " | ".join(str(c) for c in row) + " |" for row in val[1:]]
            out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "Code"))
    from autodoc.rag import SYSTEM_PROMPT
    (ROOT / "Model_Prompts_Config" / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    make_diagrams()
    B = blocks()
    to_pdf(B, DOC / "Technical_Report.pdf")
    to_md(B, DOC / "Technical_Report_content.md")
    print("Created", DOC / "Technical_Report.pdf")
    print("Created", DOC / "Technical_Report_content.md")
    print("Created diagrams in", DIAG)
