# config.py
import os
import dspy

# Global system guidance to bias GPT‑4o‑mini (or others) toward crisp, safe reasoning.
SYSTEM_PERSONA = (
    "You are an ensemble orchestrator and expert panel tuned for precise, text-only reasoning. "
    "Rules: (1) Be concise but explicit; (2) If uncertain, say 'unknown' and list what would verify; "
    "(3) Prefer bullets/numbered lists; (4) Separate CLAIMS/EVIDENCE/CAVEATS when useful; "
    "(5) No fabricated sources or data; (6) Obey the given mode (explore|verify|attack|plan) and goal; "
    "(7) Keep outputs under ~300 tokens unless required."
)

def _provider_requires_key(model: str) -> bool:
    prov = model.split("/", 1)[0]
    return prov not in {"ollama_chat"}  # local endpoints often don't require keys

def configure_lm():
    """
    Provider‑agnostic LM config for DSPy.
    Env (set in your shell):
      DSPY_LM        e.g., openai/gpt-4o-mini | anthropic/claude-3-haiku-20240307 | gemini/gemini-2.5-pro-preview-03-25 | ollama_chat/llama3.2
      DSPY_API_BASE  e.g., http://localhost:11434  (Ollama) or any OpenAI‑compatible base
      DSPY_API_KEY   overrides provider key if set
    Provider keys also respected: OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY / DATABRICKS_API_KEY
    """
    model = os.getenv("DSPY_LM", "openai/gpt-4o-mini")
    api_base = os.getenv("DSPY_API_BASE", None)
    api_key  = os.getenv("DSPY_API_KEY", None)

    if api_key is None:
        provider = model.split("/", 1)[0]
        by_provider_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "databricks": "DATABRICKS_API_KEY",
            "ollama_chat": None,   # no key
        }
        envvar = by_provider_env.get(provider)
        api_key = os.getenv(envvar, "") if envvar else ""

    lm_kwargs = dict(temperature=0.2, cache=True, max_tokens=1024)
    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key is not None:
        lm_kwargs["api_key"] = api_key

    if _provider_requires_key(model) and not api_key:
        raise RuntimeError(f"{model}: missing API key. Set DSPY_API_KEY or the provider key env var.")

    dspy.configure(lm=dspy.LM(model, **lm_kwargs))
