"""AutoDoc AI - AUTOSAR HLD Document Analysis Assistant (Streamlit).

Run from the project root:   streamlit run Code/app.py
No LLM, no API keys, no internet needed after installation. Retrieval = TF-IDF + cosine similarity.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from autodoc.checks import RULES, run_checks
from autodoc.compare import compare_docs
from autodoc.config import CFG, INPUT_DIR
from autodoc.exports import to_csv_bytes, to_json_bytes
from autodoc.extract import build_inventory, build_relationships, flow_summary
from autodoc.ingest import file_hash, ingest_pdf, ocr_available
from autodoc.embeddings import EmbeddingUnavailable, SentenceTransformerEmbedder
from autodoc.llm import LLMClient
from autodoc.rag import RagEngine
from autodoc.retrieval import Index, answer

st.set_page_config(page_title="AutoDoc AI", layout="wide")

NOTICE = ("**Responsible-use notice.** AutoDoc AI is a decision-support prototype, not an AUTOSAR compliance "
          "checker and not a replacement for engineering review. Answers are *extractive* (text copied from "
          "your PDFs - there is no generative model). Extracted items are *candidates* and may be wrong or "
          "incomplete (false positives and misses, especially for tables and diagrams). Confidence values are "
          "heuristics, not probabilities. A qualified engineer must verify everything before it is used for "
          "design approval or release. Do not upload confidential documents to a shared/cloud deployment.")


@st.cache_data(show_spinner="Extracting text page by page...")
def cached_ingest(name: str, data: bytes, use_ocr: bool):
    return ingest_pdf(name, data, use_ocr)


@st.cache_resource(show_spinner="Building TF-IDF index...")
def cached_index(key: tuple, _docs):
    return Index(_docs)


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB once)...")
def get_embedder(name: str):
    try:
        return SentenceTransformerEmbedder(name), ""
    except EmbeddingUnavailable as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Embedding passages and building the vector index...")
def get_engine(key: tuple, _docs, _embedder, embed_err: str):
    return RagEngine(_docs, embedder=_embedder, llm=LLMClient(), embed_error=embed_err)


@st.cache_data(ttl=20, show_spinner=False)
def llm_health(
    provider: str, url: str, model: str, api_key: str = ""
):
    return LLMClient(
        provider, url, model, api_key or None
    ).health()

@st.cache_data(show_spinner="Extracting candidate architecture elements...")
def cached_knowledge(key: tuple, _docs):
    rows, per_sentence = build_inventory(_docs)
    return rows, build_relationships(per_sentence), run_checks(per_sentence)


# ---------------------------------------------------------------- sidebar: input
st.title("AutoDoc AI - AUTOSAR HLD Document Analysis Assistant")
st.caption("Case Study 1 prototype | offline | evidence-grounded | human review required")
with st.expander("Responsible-use notice (read first)", expanded=False):
    st.markdown(NOTICE)

sources = {}  # name -> bytes
with st.sidebar:
    st.header("1. Documents")
    uploads = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    for f in uploads or []:
        sources[f.name] = f.getvalue()
    local = sorted(INPUT_DIR.glob("*.pdf")) if INPUT_DIR.exists() else []
    DEMO = ["SYNTHETIC_Wiper_HLD_v1.0.pdf", "SYNTHETIC_Wiper_HLD_v1.1.pdf"]
    if all((INPUT_DIR / n).exists() for n in DEMO):
        st.button("Load synthetic demo documents", on_click=lambda: st.session_state.update(picked=DEMO))
    picked = st.multiselect("...or pick from Input_Data/", [p.name for p in local], key="picked")
    for n in picked:
        sources[n] = (INPUT_DIR / n).read_bytes()
    st.header("2. AI settings")
    L = CFG["llm"]
    prov_default = (__import__("os").environ.get("AUTODOC_LLM_PROVIDER") or L["provider"]).lower()
    provs = ["ollama", "openai_compatible", "none"]
    provider = st.selectbox("LLM provider", provs, index=provs.index(prov_default) if prov_default in provs else 0,
                            help="ollama = local model via the Ollama app. openai_compatible = any hosted/local "
                                 "endpoint configured by environment variables. none = retrieval only.")
    cur = LLMClient(provider)
    model = st.text_input("Model", value=cur.model)
    base_url = st.text_input("Base URL", value=cur.base_url)
    api_key = st.text_input("API key (optional, session only)", type="password",
                            help="Better: set the AUTODOC_LLM_API_KEY environment variable. Never saved to disk.")
    min_score = st.slider("Minimum similarity for evidence", 0.0, 0.8, float(CFG["rag"]["dense_min_score"]), 0.01,
                          help="Below this the app refuses to answer. Tune with your documents.")
    if st.button("Retry loading embedding model"):
        get_embedder.clear()
        get_engine.clear()
    st.header("3. Options")
    ocr_ok = ocr_available()
    use_ocr = st.checkbox("Use OCR for scanned pages", value=False, disabled=not ocr_ok,
                          help="Needs pytesseract, Pillow and the Tesseract program installed.")
    if not ocr_ok:
        st.caption("OCR not available (optional). Searchable PDFs work without it.")
    st.caption("Everything runs locally. Uploaded files are held in memory only.")

docs = [cached_ingest(n, b, use_ocr) for n, b in sources.items()]
usable = [d for d in docs if d["status"] in ("ok", "partial")]
key = tuple(sorted((d["name"], d["hash"], use_ocr) for d in usable))

llm = LLMClient(provider, base_url or None, model or None, api_key or None)
embedder, embed_err = get_embedder(CFG["rag"]["embedding_model"]) if usable else (None, "no documents loaded")
engine = None
if usable:
    engine = get_engine(key, usable, embedder, embed_err)
    engine.llm = llm  # provider can change without re-embedding

# ---------------------------------------------------------------- health / status panel
with st.container(border=True):
    st.markdown("**System status**")
    h = llm_health(
    llm.provider, llm.base_url, llm.model, api_key
) if llm.configured else {"ok": False, "detail": "none"}
    st_ = engine.status() if engine else {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", len(usable))
    c2.metric("LLM", llm.label() if llm.configured else "none", "reachable" if h["ok"] else "NOT reachable",
              delta_color="normal" if h["ok"] else "inverse")
    c3.metric("Embeddings", (st_.get("embedding_model") or "unavailable").split("/")[-1])
    c4.metric("Index", f"{st_.get('chunks', 0)} chunks", st_.get("vector_backend", "none"), delta_color="off")
    genuine = bool(engine and engine.store is not None and llm.configured and h["ok"])
    c5.metric("Answer mode", "LLM + RAG" if genuine else "FALLBACK", None)
    if genuine:
        st.success("Genuine LLM + RAG is active (semantic embeddings + LLM generation from retrieved passages).")
    else:
        why = []
        if usable and (engine is None or engine.store is None):
            why.append(f"embeddings unavailable ({embed_err[:160]})")
        if not llm.configured:
            why.append("no LLM provider configured")
        elif not h["ok"]:
            why.append(h["detail"][:200])
        if not usable:
            why.append("no documents loaded yet")
        st.warning("Fallback mode - answers will be clearly labelled. Reason: " + "; ".join(why or ["unknown"]))
    if engine:
        st.caption(f"Index cache: {st_.get('index_cache')}")

tabs = st.tabs(["Documents", "Ask (Q&A)", "Architecture knowledge", "Functional flows",
                "Change Impact Radar", "Potential inconsistencies", "Exports & review"])

# ---------------------------------------------------------------- Documents
with tabs[0]:
    if not docs:
        st.info("Upload a PDF or pick one from Input_Data/ in the sidebar. "
                "Generate sample files first with: python Code/generate_synthetic.py")
    for d in docs:
        icon = {"ok": "OK", "partial": "PARTIAL"}.get(d["status"], "PROBLEM")
        st.markdown(f"**{d['name']}** - {icon} - status: `{d['status']}` - pages: {d['page_count']} - "
                    f"content hash (version check): `{d['hash']}`")
        if d["message"]:
            (st.warning if d["status"] in ("partial", "no_text", "encrypted") else st.error)(d["message"])
        if d["pages"]:
            st.caption(f"Pages with little/no text: {d['low_text_pages'] or 'none'} | OCR pages: "
                       f"{d['ocr_pages'] or 'none'} | repeated header/footer lines removed: "
                       f"{d['removed_header_footer_lines']} | headings detected: {len(d['headings'])}")
            with st.expander("Page-level extraction table"):
                st.dataframe(pd.DataFrame([{"page": p["page"], "characters": p["chars"], "source": p["source"]}
                                           for p in d["pages"]]), hide_index=True)
    if usable:
        idx = cached_index(key, usable)
        st.success(f"Indexed {len(idx.chunks)} passages (page-level source metadata preserved).")

if not usable:
    for t in tabs[1:]:
        with t:
            st.info("No usable document loaded yet.")
    st.stop()

index = cached_index(key, usable)
inventory, relationships, findings = cached_knowledge(key, usable)

# ---------------------------------------------------------------- Q&A
with tabs[1]:
    st.subheader("Ask a question about the loaded documents")
    method = st.radio("Method", ["RAG: semantic retrieval + LLM (recommended)", "Legacy: TF-IDF extractive (no LLM)"],
                      horizontal=True)
    scope = st.multiselect("Restrict to documents (optional)", [d["name"] for d in usable])
    q = st.text_input("Question", placeholder="e.g. Which port is provided by RainSensorSwc?")
    ask = st.button("Ask", type="primary") or bool(q and st.session_state.get("last_q") == q)
    if q and ask:
        st.session_state["last_q"] = q
        if method.startswith("RAG"):
            with st.spinner("Retrieving evidence and generating a grounded answer (CPU models can take 10-60 s)..."):
                res = engine.ask(q, min_score=min_score, doc_filter=set(scope) or None)
            badge = {"llm_rag": st.success, "retrieval_only": st.warning, "lexical_fallback": st.warning,
                     "refused": st.info}[res["mode"]]
            badge("Answer mode: " + res["mode_label"])
            for w in res["warnings"]:
                st.warning(w)
            if res["status"] == "insufficient":
                st.info(res["message"] or "Insufficient evidence.")
            elif res["mode"] == "llm_rag":
                st.markdown("**Answer (generated by the LLM from the retrieved passages only):**")
                st.write(res["answer"])
                if res["citations"]:
                    st.markdown("**Citations:** " + "; ".join(
                        f"[{c['tag']}] {c['doc']}, page {c['page']}" for c in res["citations"]))
                st.caption("Verify every statement against the evidence below. LLM output can still be wrong.")
            elif res["points"]:
                st.markdown("**Retrieved excerpts (nothing generated):**")
                for pt in res["points"]:
                    st.markdown(f"- {pt['text']}  \n  *Source: {pt['doc']}, page {pt['page']}*")
            if res["evidence"]:
                st.markdown("**Retrieved evidence (passages sent to the LLM are [S1], [S2], ... in this order):**")
                for i, h_ in enumerate(res["evidence"], 1):
                    with st.expander(f"[S{i}] {h_['doc']} - page {h_['page']} - similarity {h_['score']:.2f}",
                                     expanded=(i == 1)):
                        st.write(h_["text"])
                        st.caption(f"Section: {h_['section'] or 'n/a'}")
            st.session_state["last_evidence"] = [
                {"question": q, "answer_mode": res["mode"], "answer": res["answer"], "document": h_["doc"],
                 "page": h_["page"], "section": h_["section"], "score": round(h_["score"], 3),
                 "excerpt": h_["text"]} for h_ in res["evidence"]]
        else:
            res = answer(index, q, doc_filter=scope or None)
            st.warning("Answer mode: LEGACY TF-IDF extractive - lexical matching, NOT semantic, NOT an LLM.")
            c1, c2 = st.columns(2)
            c1.metric("Relevance heuristic", res["confidence"])
            c2.metric("Top cosine score", f"{res['top_score']:.2f}")
            if res["status"] == "insufficient":
                st.info(res["message"])
            else:
                for pt in res["points"]:
                    st.markdown(f"- {pt['text']}  \n  *Source: {pt['doc']}, page {pt['page']}*")
            for i, h_ in enumerate(res["evidence"], 1):
                with st.expander(f"{i}. {h_['doc']} - page {h_['page']} - score {h_['score']:.2f}", expanded=(i == 1)):
                    st.write(h_["text"])
            st.session_state["last_evidence"] = [
                {"question": q, "answer_mode": "legacy_tfidf", "answer": "", "document": h_["doc"],
                 "page": h_["page"], "section": h_["section"], "score": round(h_["score"], 3),
                 "excerpt": h_["text"]} for h_ in res["evidence"]]

# ---------------------------------------------------------------- Knowledge
with tabs[2]:
    st.subheader("Candidate architecture inventory")
    st.warning("Automatically extracted CANDIDATES from naming patterns. They require human verification. "
               "Nothing is invented: every row cites its first page and sentence.")
    inv_df = pd.DataFrame(inventory)
    if inv_df.empty:
        st.info("No candidate elements matched the extraction patterns in this document.")
    else:
        types = st.multiselect("Filter by type", sorted(inv_df["type"].unique()),
                               default=sorted(inv_df["type"].unique()))
        view = inv_df[inv_df["type"].isin(types)].copy()
        view["review_status"] = "Unreviewed"
        view["reviewer_note"] = ""
        edited = st.data_editor(
            view, hide_index=True, key="inv_editor",
            column_config={"review_status": st.column_config.SelectboxColumn(
                "review_status", options=["Unreviewed", "Confirmed", "Rejected", "Needs discussion"])},
            disabled=[c for c in view.columns if c not in ("review_status", "reviewer_note")])
        st.session_state["reviewed_inventory"] = edited.to_dict("records")
    st.subheader("Possible relationships (candidates)")
    st.caption("Rule: two named elements in one sentence + a verb cue (sends, calls, provides...). "
               "Direction follows word order and may be wrong.")
    rel_df = pd.DataFrame(relationships)
    if rel_df.empty:
        st.info("No candidate relationships found.")
    else:
        st.dataframe(rel_df, hide_index=True)

# ---------------------------------------------------------------- Flows
with tabs[3]:
    st.subheader("Functional-flow summary from retrieved passages")
    topic = st.text_input("Flow topic", value="", placeholder="e.g. startup, fault handling, diagnostic request")
    if topic:
        fs = flow_summary(index, None, topic)
        if fs["status"] == "insufficient":
            st.warning("Not enough evidence in the loaded documents for this topic.")
        else:
            st.markdown("**Sections involved:** " + ", ".join(fs["sections"] or ["n/a"]))
            st.markdown("**Extracted evidence (verbatim, cited):**")
            for s in fs["summary"]:
                st.markdown(f"- {s['text']}  \n  *{s['doc']}, page {s['page']}*")
            st.info("**Inferred (rule-based) flow:** the sentences above, listed in document order. "
                    "Document order is NOT guaranteed to equal execution order - verify against the source.")
            st.session_state["flow_export"] = [{"topic": topic, "step": i + 1, "document": s["doc"],
                                                "page": s["page"], "text": s["text"]}
                                               for i, s in enumerate(fs["summary"])]

# ---------------------------------------------------------------- Compare
with tabs[4]:
    st.subheader("Change Impact Radar")
    names = [d["name"] for d in usable]
    if len(names) < 2:
        st.info("Load at least two documents (e.g. SYNTHETIC_Wiper_HLD_v1.0.pdf and ..._v1.1.pdf).")
    else:
        a = st.selectbox("Older version", names, index=0)
        b = st.selectbox("Newer version", names, index=1)
        if a == b:
            st.warning("Choose two different documents.")
        else:
            da = next(d for d in usable if d["name"] == a)
            db = next(d for d in usable if d["name"] == b)
            rows, summ, impact = compare_docs(da, db)
            st.session_state["compare_rows"], st.session_state["impact_rows"] = rows, impact
            m1, m2, m3 = st.columns(3)
            m1.metric("Added", summ["added"])
            m2.metric("Removed", summ["removed"])
            m3.metric("Modified", summ["modified"])
            st.caption("Sentence-level text comparison. A change is not necessarily a defect or a safety issue.")
            st.markdown("**Elements mentioned in changed text (flag for review):**")
            st.dataframe(pd.DataFrame(impact), hide_index=True)
            for r in rows:
                label = f"{r['change_type'].upper()} - section {r['section']} - p.{r['before_page'] or '-'} -> p.{r['after_page'] or '-'}"
                with st.expander(label):
                    c1, c2 = st.columns(2)
                    c1.markdown("**Before**")
                    c1.write(r["before_text"] or "(none)")
                    c2.markdown("**After**")
                    c2.write(r["after_text"] or "(none)")
                    if r["word_level_change"]:
                        st.caption(r["word_level_change"])
                    st.caption("Potentially affected: " + (r["potentially_affected_elements"] or "none detected"))

# ---------------------------------------------------------------- Checks
with tabs[5]:
    st.subheader("Potential inconsistency report")
    st.warning("Four simple, explainable rules. Findings are candidates for review. This is NOT formal AUTOSAR "
               "compliance verification.")
    with st.expander("Rules and limitations"):
        for k, v in RULES.items():
            st.markdown(f"- **{k}**: {v}")
        st.markdown("- Limitations: pattern-based; tables/diagrams may be mis-read; synonyms and abbreviations "
                    "are not resolved; absence of a finding does not mean the document is consistent.")
    if findings:
        st.dataframe(pd.DataFrame(findings)[["rule", "finding", "evidence", "pages", "status"]],
                     hide_index=True)
    else:
        st.success("No findings from the four rules (this does not prove consistency).")

# ---------------------------------------------------------------- Exports
with tabs[6]:
    st.subheader("Downloads (CSV / JSON)")
    st.caption("Exports include page references and 'candidate / needs review' status columns.")
    exports = {
        "inventory_reviewed": st.session_state.get("reviewed_inventory", inventory),
        "relationships": relationships,
        "inconsistencies": findings,
        "retrieved_evidence": st.session_state.get("last_evidence", []),
        "flow_summary": st.session_state.get("flow_export", []),
        "comparison": st.session_state.get("compare_rows", []),
        "change_impact": st.session_state.get("impact_rows", []),
    }
    for name, rows in exports.items():
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.write(f"**{name}** - {len(rows)} rows")
        if rows:
            c2.download_button("CSV", to_csv_bytes(rows), f"{name}.csv", "text/csv", key=f"c_{name}")
            c3.download_button("JSON", to_json_bytes(rows), f"{name}.json", "application/json", key=f"j_{name}")
        else:
            c2.caption("(empty)")
    st.divider()
    st.markdown(NOTICE)
    st.caption(f"Active configuration: {CFG['retrieval']}")
