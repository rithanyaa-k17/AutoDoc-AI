"""Loads Model_Prompts_Config/config.json (falls back to built-in defaults)."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "Model_Prompts_Config" / "config.json"
INPUT_DIR = PROJECT_ROOT / "Input_Data"
RESULTS_DIR = PROJECT_ROOT / "Evaluation_Results"

DEFAULTS = {
    "retrieval": {"chunk_max_words": 90, "top_k": 5, "min_score": 0.08,
                  "min_query_coverage": 0.34, "confidence_high": 0.35, "confidence_medium": 0.18,
                  "chunk_overlap_sentences": 1},
    "ingestion": {"low_text_chars": 30, "repeat_line_fraction": 0.6, "repeat_line_min_pages": 4},
    "extraction": {
        "component_suffixes": ["Swc", "SWC", "Ctrl", "Controller", "Manager", "Driver", "Handler",
                              "Monitor", "Provider", "Consumer"],
        "interface_suffixes": ["Interface", "Service", "Proxy", "Skeleton", "If"],
        "title_case_component_words": ["Management", "Manager", "Cluster", "Application", "Daemon",
                                       "Gateway", "Controller", "Binding"],
        "title_case_interface_words": ["Interface"],
        "max_entities_per_sentence_for_relations": 5},
    "checks": {"near_duplicate_ratio": 0.9},
    "rag": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "dense_top_k": 5,
            "dense_min_score": 0.22, "max_context_chars": 6000, "cache_dir": ".cache_index"},
    "llm": {"provider": "ollama", "ollama_base_url": "http://localhost:11434/v1", "ollama_model": "llama3.2:3b",
            "openai_compatible_base_url": "", "openai_compatible_model": "", "temperature": 0.0,
            "max_tokens": 500, "timeout_s": 180},
}


def load_config():
    cfg = json.loads(json.dumps(DEFAULTS))
    try:
        user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for section, values in user.items():
            if isinstance(values, dict) and section in cfg:
                cfg[section].update(values)
    except (OSError, ValueError):
        pass  # keep defaults
    return cfg


CFG = load_config()
