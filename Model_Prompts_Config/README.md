# Models, prompt and configuration
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (override: env `AUTODOC_EMBED_MODEL`). Weights are downloaded to the Hugging Face cache on your PC, not stored in this project.
- **LLM:** none bundled. Provider/model are configured by sidebar, `config.json` (`llm` section, no secrets) or env vars `AUTODOC_LLM_PROVIDER`, `_BASE_URL`, `_MODEL`, `_API_KEY`.
- **System prompt:** `system_prompt.txt` (exported from `Code/autodoc/rag.py` by `generate_report.py`; the code is the single source of truth). Context is supplied as numbered `[S#]` passages with file/page/section.
- **`config.json`:** chunk size/overlap, top-k, `dense_min_score` (untuned default 0.22), cache folder, LLM temperature/timeouts, extraction suffix lists. Re-run `evaluate_rag.py` after changing.
- The TF-IDF fallback and rule-based extraction patterns live in `Code/autodoc/retrieval.py` / `extract.py`.
