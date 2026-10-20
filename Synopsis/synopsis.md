# Synopsis - AutoDoc AI

**Student:** Rithanyaa K | **Register No:** 127003212 | **Institution:** SASTRA Deemed University | **Department:** B.Tech Computer Science & Engineering | **Case study:** CS1 - AUTOSAR HLD Document Analysis Assistant | **Date:** 14 October 2026

**Problem.** AUTOSAR HLD documents are long unstructured PDFs; finding components, interfaces, flows and version changes is slow and inconsistent.

**Proposed solution.** A Streamlit prototype that ingests PDFs page by page and answers questions with a RAG pipeline: sentence-transformer embeddings (all-MiniLM-L6-v2), a local vector store (FAISS or NumPy), an evidence-only prompt and a configurable LLM (Ollama locally, or a hosted OpenAI-compatible model such as `openai/gpt-oss-20b` via Groq), with page-cited answers, visible evidence and labelled fallbacks (retrieval-only, TF-IDF extractive, refusal). It also extracts *candidate* components/interfaces/ports/signals, compares two versions (Change Impact Radar), reports *potential* inconsistencies, and exports CSV/JSON with a human-review column.

**Inputs.** Two fictional synthetic HLDs (`SYNTHETIC_Wiper_HLD_v1.0/v1.1.pdf`) with planted issues and revisions; users may upload their own non-confidential PDFs. No real AUTOSAR or Tata Technologies documents are included.

**Responsible AI.** Grounded prompt, `[S#]` citations, similarity gate, visible evidence, human review. With a hosted LLM the retrieved passages leave the machine, so only synthetic or public documents should be used.

**Deliverables.** Technical report (PDF + DOCX), 5-10 min demonstration video, signed integrity and AI-tool declarations, this folder structure and ZIP.
