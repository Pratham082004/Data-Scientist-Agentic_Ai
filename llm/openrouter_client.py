"""
OpenRouter LLM Client (Stable JSON Mode for Hybrid AutoML)
----------------------------------------------------------

Upgraded features:
✔ Uses your model: meta-llama/llama-3.3-70b-instruct:free
✔ Unlimited tokens via max_output_tokens
✔ Safe JSON extraction
✔ Retries for rate-limit / server errors
✔ Non-streaming (required for JSON reliability)
"""

from __future__ import annotations
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OpenRouterClient:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        api_key: Optional[str] = None,
    ):
        # Choose model in correct priority order
        self.model = (
            model
            or os.getenv("OPENROUTER_MODEL")
            or "meta-llama/llama-3.3-70b-instruct:free"
        )

        self.base_url = base_url

        # Secure key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

        # OpenRouter required headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://data-scientist-agentic-ai",
            "X-Title": "Data Scientist Agentic AI",
        }

    # --------------------------------------------------------------
    # SAFE JSON EXTRACTOR
    # --------------------------------------------------------------
    def extract_json(self, text: str):
        """Extracts JSON from any LLM output."""
        try:
            return json.loads(text)
        except:
            pass

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except:
            logger.warning("JSON extraction failed. Returning raw text.")
            return text

    # --------------------------------------------------------------
    # MAIN CHAT COMPLETION CALL
    # --------------------------------------------------------------
    def chat_completion(
        self,
        prompt: str,
        role: str = "user",
        system_prompt: Optional[str] = None,
        max_tokens: int = 128000,     # unlimited-style
        temperature: float = 0.2,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        retries: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:

        # Build messages array
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if extra_messages:
            messages.extend(extra_messages)

        messages.append({"role": role, "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens,   # correct OpenRouter key
        }
        payload.update(kwargs)

        # Retry loop
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=90,
                )

                if response.status_code == 429:
                    logger.warning("Rate limited. Retry %d/%d", attempt, retries)
                    time.sleep(2 * attempt)
                    continue

                response.raise_for_status()
                data = response.json()
                return data

            except Exception as e:
                msg = response.text if "response" in locals() else "NO RESPONSE"
                logger.error(f"OpenRouter Error (attempt {attempt}/{retries}): {e} | {msg}")
                time.sleep(1.5 * attempt)

        raise RuntimeError("OpenRouter LLM failed after retries")

    # --------------------------------------------------------------
    # SIMPLE “ASK” API FOR AGENTS WANTING JSON DIRECTLY
    # --------------------------------------------------------------
    def ask(self, prompt: str, system_prompt: Optional[str] = None):
        resp = self.chat_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=128000,
            temperature=0.1,
        )
        content = resp["choices"][0]["message"]["content"]
        return self.extract_json(content)
