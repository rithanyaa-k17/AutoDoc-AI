# Deployment
**Primary: local Windows** (`streamlit run Code\\app.py`) with Ollama - the only configuration that keeps everything on your PC.
Free hosting (e.g. Streamlit Community Cloud) is NOT recommended for the genuine local-LLM demo: it cannot run Ollama, PyTorch/sentence-transformers may exceed free memory limits, and documents would leave your machine. If you try it, use synthetic documents only, add `AUTODOC_LLM_*` as the platform's secrets (never in Git), and expect fallback modes. Unverified.
