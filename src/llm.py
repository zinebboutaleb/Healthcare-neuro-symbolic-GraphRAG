"""LLM abstraction. LLM_PROVIDER = auto | anthropic | openai | ollama | hf | offline

  auto      -> anthropic if ANTHROPIC_API_KEY, else openai if OPENAI_API_KEY, else offline
  hf        -> local open-source Hugging Face model (LLM_MODEL, default Qwen/Qwen2.5-1.5B-Instruct)
  offline   -> NO model: returns the retrieved context verbatim (an "extractive echo"). It lets the whole
               pipeline and the retrieval evaluation run without keys. It measures *what was retrieved*,
               not generation quality - an LLM is needed for real answer-quality / hallucination numbers.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import env


@lru_cache(maxsize=1)
def _load_hf(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return tok, model, device


def _hf_generate(model_name: str, system: str, user: str) -> str:
    import torch

    tok, model, device = _load_hf(model_name)
    text = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(env("HF_MAX_NEW_TOKENS", 250)), do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

SYSTEM_PROMPT = (
    "You are a research assistant for a SYNTHETIC, educational clinical knowledge base (not medical advice).\n"
    "Answer the question using ONLY the supplied context. Do not use outside medical knowledge.\n"
    "If the context is insufficient, say exactly what is missing instead of guessing.\n"
    "Be concise (max ~120 words). Name the specific entities in your answer and cite the evidence tags "
    "you used, e.g. [D1], [G3], [R2]."
)


def build_user_prompt(question: str, context: str) -> str:
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER (grounded in the context only):"


class LLM:
    def __init__(self, provider: str | None = None):
        provider = (provider or env("LLM_PROVIDER", "auto")).lower()
        if provider == "auto":
            provider = "anthropic" if env("ANTHROPIC_API_KEY") else "openai" if env("OPENAI_API_KEY") else "offline"
        self.provider = provider
        self.model = {"anthropic": env("LLM_MODEL", "claude-sonnet-5-5"), "openai": env("LLM_MODEL", "gpt-4o-mini"),
                      "ollama": env("LLM_MODEL", "llama3.1:8b"),
                      "hf": env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"), "offline": "none"}.get(provider, "?")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def answer(self, question: str, context: str) -> str:
        prompt = build_user_prompt(question, context)
        if self.provider == "offline":
            return "[offline extractive mode - no LLM configured; showing the retrieved context]\n" + context
        if self.provider == "anthropic":
            import anthropic

            msg = anthropic.Anthropic().messages.create(model=self.model, max_tokens=700, temperature=0, system=SYSTEM_PROMPT,
                                                        messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if b.type == "text").strip()
        if self.provider == "openai":
            from openai import OpenAI

            r = OpenAI().chat.completions.create(model=self.model, temperature=0, messages=[
                {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
            return r.choices[0].message.content.strip()
        if self.provider == "ollama":
            import requests

            r = requests.post(f"{env('OLLAMA_URL', 'http://localhost:11434')}/api/chat", timeout=300, json={
                "model": self.model, "stream": False, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        if self.provider == "hf":
            return _hf_generate(self.model, SYSTEM_PROMPT, prompt)
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")