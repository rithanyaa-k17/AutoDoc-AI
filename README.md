# AutoDoc AI - AUTOSAR HLD Document Analysis Assistant (local RAG)

Case Study 1 prototype. **Pipeline:** PyMuPDF -> overlapping chunks (file/page/section metadata) -> sentence-transformers
embeddings (`all-MiniLM-L6-v2`) -> local vector store (FAISS or NumPy) -> controlled prompt -> **LLM** -> answer with
`[S#]` citations + visible evidence. If embeddings or the LLM are unavailable the app **says so** and uses a labelled
fallback; it never presents TF-IDF or excerpts as LLM output. Decision support only - not an AUTOSAR compliance checker.

**Honest status:** all code was tested in a Linux sandbox (Python 3.12) with test doubles (stub embedder, mock LLM server).
**Not verified by the author:** Windows 11 / Python 3.13 install, real MiniLM download, real Ollama/hosted LLM answers.
`check_setup.py` and `evaluate_rag.py` tell you on your machine.

## 1. One-time setup (PowerShell, needs internet once)
```powershell
cd D:\TechPulse\AutoDoc-AI
.\.venv\Scripts\Activate.ps1             # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Code\generate_synthetic.py
python Code\check_setup.py --download    # downloads MiniLM (~90 MB) once and tests it
```
(or run `.\setup_windows.ps1`). Optional FAISS: `python -m pip install -r requirements-optional.txt` (if it fails, ignore: NumPy search is used).

## 2. Run
```powershell
streamlit run Code\app.py                # http://localhost:8501 ; click "Load synthetic demo documents"
```
The **System status** panel must show *Answer mode: LLM + RAG* (green). Otherwise it shows why (fallback).

## 3. Choose your LLM (pick ONE)
**A. Local - Ollama (nothing leaves your PC; ~2 GB model, CPU is fine, slower).**
1. Install Ollama for Windows from https://ollama.com/download and start it.
2. `ollama pull llama3.2:3b`   (faster/smaller: `ollama pull qwen2.5:1.5b`, then set Model in the sidebar or `$env:AUTODOC_LLM_MODEL="qwen2.5:1.5b"`).
3. Nothing else: provider `ollama` and `http://localhost:11434/v1` are the defaults.

**B. Hosted OpenAI-compatible endpoint (fastest to try; sends retrieved passages to the provider -> synthetic/public docs ONLY).**
```powershell
$env:AUTODOC_LLM_PROVIDER = "openai_compatible"
$env:AUTODOC_LLM_BASE_URL = "<provider base URL ending in /v1>"
$env:AUTODOC_LLM_MODEL    = "<model name from the provider docs>"
$env:AUTODOC_LLM_API_KEY  = "<your key>"      # session only; never put keys in files or Git
streamlit run Code\app.py
```
Free tiers exist at some providers (check their current terms). Any local OpenAI-compatible server (LM Studio, llama.cpp server) also works with provider B.

## 4. Tests (commands)
```powershell
python -m unittest discover -s Code\tests -v   # deterministic OFFLINE tests (no model, no key)
python Code\evaluate.py                        # synthetic evaluation of analysis modules -> Evaluation_Results\
python Code\evaluate_rag.py                    # REAL embeddings + REAL LLM integration run (needs setup + LLM)
python Code\generate_report.py                 # technical report PDF (run AFTER the evaluations)
python Code\package_zip.py --name UniversityName_StudentName_RegisterNo_CSx_AIML
```
See `Documentation/testing_guide.md` for what each test does and does not prove.

## 5. Implemented vs planned
Implemented: ingestion, overlapping chunking, MiniLM embeddings, FAISS/NumPy store with disk cache, Ollama/OpenAI-compatible LLM,
grounded prompt, citation validation, refusal gate, status panel, answer-mode labels, entity extraction, flows, version comparison,
inconsistency rules, CSV/JSON export, review column. **Not implemented:** FastAPI, SQLite/PostgreSQL, graph store, RBAC, audit log,
in-process GGUF model, reliable OCR, table-structure extraction. Folder map: Synopsis, Input_Data, Code, Model_Prompts_Config,
Evaluation_Results, Documentation, Video, Declarations.
