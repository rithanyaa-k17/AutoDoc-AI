# Still unverified (state as "Not tested" / "Not measured" in the submission)

| Item | Status | How to close it |
|---|---|---|
| Real all-MiniLM-L6-v2 retrieval quality | Not measured | `python Code\check_setup.py --download`, then `python Code\evaluate_rag.py` |
| `dense_min_score` = 0.22 calibration | Not measured | Run in/out-of-scope questions, adjust slider, re-run `evaluate_rag.py` |
| Correctness of LLM answers (incl. the `phi:latest` demo answer) | Not tested | Fill `STUDENT_VERDICT` in `rag_llm_results.csv`; compare each statement with [S1]-[S5] |
| Citation correctness (support, not just tag existence) | Not measured | Manual review per answer |
| Small-model citation-format compliance (`phi`) | Not measured | Count answers with valid `[S#]` in `rag_llm_summary.json` |
| Prompt-injection resistance | Not tested | Add adversarial text to a synthetic page and re-ask |
| FAISS vs NumPy backend in your live session | Not established | Read "Index" metric in the status panel screenshot |
| Windows 11 / Python 3.13 clean install, PyTorch install | Not tested by author | Follow README on a clean venv; record versions |
| Real or non-synthetic HLD / AUTOSAR PDF | Not tested | Add a non-confidential PDF; manually check cited pages |
| OCR for scanned pages | Not tested | Install Tesseract, tick OCR on a scanned PDF |
| UI export download buttons, all seven tabs | Not tested by author | Click through using `Documentation/demo_script.md` |
| Latency, memory, scalability | Not measured | Time ingestion/embedding/generation |
| Prompt-budget edge case (>6,000 chars) | Not tested | Lower `max_context_chars` and ask a question |
| Streamlit headless run claimed in `verification_status.md` | Not re-run by author | Optional: re-run |
| Hosted `openai/gpt-oss-20b` session (answer text, correctness) | Not assessed by author | Compare each answer with its [S#] evidence; keep screenshot |
| User-Agent fix against the hosted endpoint | Not tested by author | Ask one question with provider `openai_compatible`; status panel must show LLM + RAG |
| `evaluate_rag.py` with the hosted model | Not run | Set provider env vars, run, fill `STUDENT_VERDICT` |
