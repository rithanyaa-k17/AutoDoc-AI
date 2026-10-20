# Viva questions (concise answers)
1. **What does the tool do?** Ingests HLD PDFs, answers questions with LLM-generated, page-cited (`[S#]`) answers grounded in retrieved passages (with the evidence shown), extracts candidate architecture elements, compares versions, flags potential inconsistencies, exports CSV/JSON.
2. **Is it RAG?** Yes: retrieval (MiniLM embeddings + vector search) then generation by an LLM (Ollama or an OpenAI-compatible endpoint) from the retrieved passages only. If embeddings/LLM are unavailable it shows a labelled fallback.
3. **Why Ollama / a provider instead of loading a GGUF in Python?** llama-cpp-python needs compiler/wheel luck on Windows + Python 3.13; Ollama gives local llama.cpp inference with no build step. Any OpenAI-compatible server works.
4. **What is an embedding / why cosine?** A vector capturing meaning (MiniLM, 384-d); normalised vectors make the dot product equal cosine similarity, so paraphrases land close together. TF-IDF (kept only as fallback) matches words, not meaning.
5. **How is the answer kept grounded?** System prompt restricts to numbered passages, requires `[S#]` citations (validated afterwards), exact INSUFFICIENT_EVIDENCE reply, temperature 0, and a similarity gate that skips the LLM.
6. **Chunking choices?** <=90 words, sentence-aligned, 1-sentence overlap, never across pages so every chunk has one page citation.
7. **How do you avoid unsupported answers?** Minimum-similarity gate (adjustable, untuned default), the model's own refusal, citation validation and warnings. None is perfect - a small model can still err.
8. **How are citations produced?** Every chunk keeps document name, page and section from extraction; chunks never span pages.
9. **How does extraction work?** Regular expressions on names (Swc/Interface suffixes, P_/R_ ports, Sig_ signals, Title Case "... Management", IDs). Output = candidates with page and sentence.
10. **Why only candidates?** Patterns give false positives/misses; a human must verify.
11. **How does the Change Impact Radar work?** difflib aligns sentences of two versions; unmatched = added/removed; similar pairs = modified; entities in changed text are flagged.
12. **Name two inconsistency rules and their limits.** R1 duplicate ID with different text; R4 conflicting "every N ms" for one signal. They miss paraphrased conflicts and flag legitimate repeated IDs in revision tables.
13. **How did you evaluate?** Known questions/expected pages, planted changes and inconsistencies in synthetic PDFs; unit tests. Limits: self-authored benchmark, so not evidence of real-world accuracy.
14. **FAISS vs NumPy?** Both exact inner-product search; FAISS is optional and tested equal to NumPy. At this scale NumPy is enough.
15. **What about scanned PDFs?** Pages with almost no text are flagged; OCR works only if Tesseract is installed (optional, untested here).
16. **Privacy?** Local embeddings/index; Ollama is local; hosted endpoint sends passages out - synthetic/public docs only; keys only in env vars.
17. **What would you improve?** Tune thresholds, hybrid retrieval + re-ranking, table extraction, FastAPI + SQLite review log, RBAC/audit, larger evaluation on authorised HLDs.
18. **Which AI tools did you use?** Exactly those in the AI-tool declaration: Claude (code and documentation assistance), all-MiniLM-L6-v2 (embeddings), Ollama `phi:latest` and the hosted `openai/gpt-oss-20b` (run-time LLMs). Explain what you verified yourself.
19. **Why did you add a User-Agent header in `llm.py`?** The hosted endpoint returned HTTP 403 for Python's default urllib agent; the header avoids it. (Untested by the report author against the hosted endpoint.)
20. **Is generation local?** Only with Ollama. With the hosted `openai/gpt-oss-20b` the question and retrieved passages go to the provider, so only synthetic documents were used.
