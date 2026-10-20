# Verification status (what was and was NOT actually run)
**Run in a Linux sandbox by the report author (13-14 Oct 2026), Python 3.12.3, real PyMuPDF 1.28.2 / scikit-learn 1.8.0 / faiss-cpu:**
- 27 offline unit tests passed, before and after the final code changes (11 `test_core` on real dependencies; 16 `test_rag` using a fake hashed embedder and a mock OpenAI-compatible server - plumbing only).
- `evaluate.py` (legacy TF-IDF / analysis modules) on the synthetic PDFs: results in `Evaluation_Results/`. It does not evaluate the embeddings or the LLM.
**User-reported (not independently inspected):** live runs of the Streamlit app with Ollama `phi:latest` and later the hosted `openai/gpt-oss-20b` (openai_compatible provider).
**NOT verified:** accuracy of generated answers, real MiniLM retrieval quality and the 0.22 threshold, citation correctness, prompt-injection resistance, real documents, OCR, FAISS on Windows, the User-Agent fix against the hosted endpoint, latency. See `AutoDoc_AI_Unverified_Tests_and_Capabilities.md`.
`evaluate_rag.py` has not been run; no `rag_llm_*` result files exist.
