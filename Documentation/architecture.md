# Architecture and workflow (diagrams: `diagrams/architecture.png`, `diagrams/workflow.png`)
```mermaid
flowchart LR
  A[Upload PDF] --> B[PyMuPDF text per page]
  B --> C[Overlapping chunks<br/>file + page + section]
  C --> D[MiniLM embeddings]
  D --> E[(Vector store<br/>FAISS or NumPy + disk cache)]
  Q[Question] --> F[Query embedding -> top-k cosine]
  E --> F --> G{similarity >= threshold?}
  G -- no --> R[REFUSED: LLM not called]
  G -- yes --> H[Controlled prompt with S1..Sn passages]
  H --> I[LLM: Ollama / OpenAI-compatible]
  I --> J[Citation check + answer-mode label]
  J --> K[Answer + evidence panel -> human review -> CSV/JSON]
  I -. LLM down .-> L[RETRIEVAL ONLY label]
  D -. embeddings down .-> M[FALLBACK TF-IDF label]
```
**Modules (Code/autodoc):** `ingest.py` (PDF), `retrieval.py` (chunking + TF-IDF fallback), `embeddings.py`, `vectorstore.py`, `llm.py`, `rag.py` (prompt, engine, citation check), `extract.py`, `compare.py`, `checks.py`, `exports.py`.
**Grounding controls:** context-only system prompt; untrusted-context rule; mandatory `[S#]` citations validated after generation; exact `INSUFFICIENT_EVIDENCE` reply; similarity gate that skips the LLM; temperature 0; evidence always displayed.
**Privacy boundary:** documents/embeddings local. LLM = Ollama on localhost (stays local) or a hosted endpoint (passages leave the machine).
**Deviations from the case-study reference stack:** no FastAPI, SQLite/PostgreSQL, graph store, RBAC, audit log; vector store is FAISS-or-NumPy rather than Chroma/Milvus.
