# Testing guide - three separate kinds of test
| Kind | Command | Needs | What it shows | What it does NOT show |
|---|---|---|---|---|
| 1. Deterministic offline | `python -m unittest discover -s Code\tests -v` | nothing (no model, key or internet) | chunking/page metadata, vector-store round trip, FAISS=NumPy, prompt contents, citation validation, refusal skips LLM, labelled fallbacks, invalid/encrypted/blank PDFs. Uses a hashed bag-of-words **test double** and a **mock** LLM server | semantic quality of MiniLM, real LLM answers |
| 1b. Legacy synthetic evaluation | `python Code\evaluate.py` | synthetic PDFs | planted changes/inconsistencies, extraction vs self-authored ground truth, TF-IDF Q&A | real-world accuracy |
| 2. LLM integration | `python Code\evaluate_rag.py` | MiniLM downloaded + reachable LLM | real pipeline on 7 synthetic questions + 3 out-of-scope: LLM used?, valid `[S#]` citation?, expected page retrieved?, refusal. Writes `rag_llm_results.csv` + `rag_llm_summary.json`; exits without writing if prerequisites are missing | answer correctness (you grade it in `STUDENT_VERDICT`), real-world accuracy |
| 3. Real-document manual validation (optional; not performed - no real document is included) | open the app with a non-confidential real PDF; `evaluate.py` can write `real_document_manual_review.csv` | real PDF | whether ingestion/retrieval/citations behave on a 176-page document - **you** compare each cited page with the PDF | - |
Note: the AUTOSAR reference PDF is not a project HLD; do not call results on it "HLD accuracy".
Tune `dense_min_score` (sidebar slider) if good questions are refused or unrelated ones are answered, then re-run kind 2.
