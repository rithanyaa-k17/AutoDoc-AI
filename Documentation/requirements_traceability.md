# Requirements traceability (CS = Case Study 1 doc, SI = submission instructions)
Status: Done / Partial / Not done. Items marked "unverified" were tested only with test doubles in a Linux sandbox; the real embedding model and LLMs were exercised only in the student's live demonstrations (user-reported, accuracy not assessed).
| ID | Requirement | Implementation | Evidence / artifact | Status |
|---|---|---|---|---|
| R01 | Document upload, controlled ingestion (CS) | `ingest.py`, sidebar, Documents tab | unit tests | Done |
| R02 | OCR for scanned pages (CS) | scanned-page detection; OCR optional (pytesseract) | tests for blank PDF | Partial (OCR untested) |
| R03 | Section/table/metadata extraction (CS) | headings, pages, hash; tables as plain text | report s8 | Partial |
| R04 | Chunk by section/context (CS) | overlapping sentence chunks, file+page+section metadata | `test_rag` chunk tests | Done |
| R05 | Embeddings (CS) | sentence-transformers MiniLM (`embeddings.py`) | `check_setup.py --download` | Done - real model **unverified** by author |
| R06 | Local vector DB (CS) | FAISS or NumPy store + disk cache (`vectorstore.py`) | store/FAISS tests | Done (not Chroma/Milvus) |
| R07 | Retrieve relevant sections (CS) | dense top-k cosine + gate | tests; `evaluate_rag.py` | Done |
| R08 | Grounded LLM answers with page citations (CS, SI) | `rag.py`, `llm.py`, `[S#]` prompt, citation check | mock-LLM tests; `evaluate_rag.py` | Done - real LLM **unverified** |
| R09 | Restrict to evidence; confidence/limitation; refuse when insufficient (CS) | similarity gate (LLM not called), INSUFFICIENT_EVIDENCE rule, mode labels | refusal tests | Done |
| R10 | Locally/privately hosted LLM (CS) | Ollama (local) or private endpoint | README s3 | Done (needs Ollama install) |
| R11 | Answer mode transparency (user req.) | status panel + per-answer mode label | UI test with stubs | Done |
| R12 | Entity extraction, interface/dependency mapping (CS) | `extract.py`, tabs | synthetic eval | Done (rule-based candidates) |
| R13 | Functional-flow summaries (CS) | Flows tab (cited sentences) | synthetic eval | Partial (not LLM-generated) |
| R14 | Revision comparison (CS) | `compare.py` | synthetic eval | Done |
| R15 | Mismatch/completeness checks (CS) | `checks.py` R1-R4 | synthetic eval | Partial |
| R16 | Export of structured findings (CS) | CSV/JSON | app | Done |
| R17 | Engineer review/correction (CS) | review_status column; no auto-approval | app | Done |
| R18 | RBAC, project isolation, audit logs, version approval, graph store, FastAPI, SQL DB (CS) | - | - | Not done (planned) |
| R19 | Technical report with the required sections (SI 1) | written from a source-code audit | `Documentation/AutoDoc_AI_Technical_Report_Corrected.pdf` (+ .docx) | Done |
| R20 | Video 5-10 min with real-time cited demo (SI 2) | `demo_script.md` | `Video/` | Student records |
| R21 | Signed integrity + AI-tool declarations (SI 3) | `Declarations/Declarations_Rithanyaa_K_127003212.docx` | signed copy to be placed in `Declarations/` | Student signs |
| R22 | Folder structure + ZIP (SI 4) | folder `SASTRA_RithanyaaK_127003212_CS1_AIML` + ZIP | ZIP | Done |
