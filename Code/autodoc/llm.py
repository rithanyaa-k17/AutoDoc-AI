"""LLM client for any OpenAI-compatible /chat/completions endpoint (standard library only, no SDK).

Providers
  ollama             local Ollama server (runs llama.cpp under the hood; CPU ok): http://localhost:11434/v1
  openai_compatible  any hosted or local OpenAI-compatible endpoint (Groq, OpenRouter, LM Studio, llama.cpp
                     server, ...). Configured ONLY via environment variables - keys are never written to disk.
  none               no LLM: the app falls back to a clearly labelled retrieval-only mode.
Environment variables: AUTODOC_LLM_PROVIDER, AUTODOC_LLM_BASE_URL, AUTODOC_LLM_MODEL, AUTODOC_LLM_API_KEY.
"""
import json
import os
import urllib.error
import urllib.request

from .config import CFG


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider=None, base_url=None, model=None, api_key=None):
        L = CFG["llm"]
        self.provider = (provider or os.environ.get("AUTODOC_LLM_PROVIDER") or L["provider"]).lower()
        if self.provider == "ollama":
            self.base_url = base_url or os.environ.get("AUTODOC_LLM_BASE_URL") or L["ollama_base_url"]
            self.model = model or os.environ.get("AUTODOC_LLM_MODEL") or L["ollama_model"]
        elif self.provider == "openai_compatible":
            self.base_url = base_url or os.environ.get("AUTODOC_LLM_BASE_URL") or L["openai_compatible_base_url"]
            self.model = model or os.environ.get("AUTODOC_LLM_MODEL") or L["openai_compatible_model"]
        else:
            self.provider, self.base_url, self.model = "none", "", ""
        self._api_key = api_key or os.environ.get("AUTODOC_LLM_API_KEY") or ""  # never logged or saved
        self.base_url = (self.base_url or "").rstrip("/")

    @property
    def configured(self) -> bool:
        return self.provider != "none" and bool(self.base_url) and bool(self.model)

    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.configured else "none"

    def _headers(self):
        h = {"Content-Type": "application/json",
             "User-Agent": "AutoDocAI/2.0"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def health(self) -> dict:
        """Quick reachability check (does NOT generate text)."""
        if not self.configured:
            return {"ok": False, "detail": "No LLM provider configured (provider=none or missing URL/model)."}
        try:
            req = urllib.request.Request(self.base_url + "/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))
            ids = [m.get("id") for m in data.get("data", [])]
            if ids and self.model not in ids and self.provider == "ollama":
                return {"ok": False, "detail": f"Server reachable but model '{self.model}' not pulled. "
                                               f"Run: ollama pull {self.model}  (available: {ids[:5]})"}
            return {"ok": True, "detail": f"Reachable: {self.base_url}"}
        except Exception as exc:
            hint = " Is Ollama running? (start the Ollama app)" if self.provider == "ollama" else ""
            return {"ok": False, "detail": f"Cannot reach {self.base_url}: {exc}.{hint}"}

    def chat(self, system: str, user: str) -> str:
        if not self.configured:
            raise LLMUnavailable("No LLM provider configured.")
        L = CFG["llm"]
        body = json.dumps({"model": self.model, "temperature": L["temperature"], "max_tokens": L["max_tokens"],
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers=self._headers(),
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=L["timeout_s"]) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            raise LLMUnavailable(f"LLM server returned HTTP {exc.code}: {exc.reason}") from exc
        except Exception as exc:
            raise LLMUnavailable(f"LLM call failed ({type(exc).__name__}): {exc}") from exc
