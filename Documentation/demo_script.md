# 5-minute demo (rehearse once; keep the app already loaded and Ollama/endpoint warmed up)
**Before:** `check_setup.py` shows no problems; start Ollama, or select `openai_compatible` with the Groq URL, model `openai/gpt-oss-20b` and your key (never show the key on camera); `streamlit run Code\app.py`; ask one throw-away question first (loads the model, avoids a cold start on camera).
| Time | Action | Say |
|---|---|---|
| 0:00 | Slide/README | Problem: long HLD PDFs, manual review. Users: architects, developers, testers |
| 0:30 | Click **Load synthetic demo documents**; point at **System status** | Synthetic fictional HLDs (not official AUTOSAR, not Tata). Panel: MiniLM embeddings, N chunks, backend, LLM reachable, **Answer mode: LLM + RAG** |
| 1:15 | `Documentation/diagrams/workflow.png` | PyMuPDF -> overlapping chunks -> embeddings -> vector store -> gate -> grounded prompt -> LLM -> citation check |
| 1:45 | Ask: **Which port is provided by RainSensorSwc?** | Generated answer, `[S1]` citation, page, open the evidence expander and verify |
| 2:30 | Ask: **What is the recipe for chocolate cake?** | REFUSED: LLM not even called; says what is missing |
| 3:00 | Ask a question the docs only partly cover (e.g. *What is the wiper blade material?*) | Model should answer INSUFFICIENT_EVIDENCE (small models may not - say so honestly) |
| 3:30 | Change Impact Radar v1.0 -> v1.1 | added/removed/modified, affected components; a change is not a defect |
| 4:00 | Inconsistencies tab, then Exports tab | R1-R4 planted issues; review column; CSV |
| 4:30 | Limitations | LLMs can hallucinate; thresholds untuned; documents are synthetic; with a hosted LLM the passages leave the machine; human review required |
**If the LLM is down on stage:** the panel turns amber ("FALLBACK") - show that the app labels it honestly, then explain. Do not claim LLM output in that state.
