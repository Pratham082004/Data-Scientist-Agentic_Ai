# llm/openrouter_client.py
"""
OpenRouter LLM Client
---------------------

Thin wrapper around OpenRouter's /chat/completions API.

- Loads environment variables using python-dotenv (.env file)
- Reads API key from: OPENROUTER_API_KEY
- Supports chat_completion() method for agents & coordinator

NOTE:
Never hardcode your API key. Store it only in .env.
"""

from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv  # ← NEW: dotenv support

# Load .env file if present
load_dotenv()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OpenRouterClient:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        api_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """
        - Model can be overridden by constructor, otherwise uses .env or openrouter.yaml
        - API key is always taken from .env unless manually provided
        """

        # Load model from env if provided
        self.model = (
            model
            or os.getenv("OPENROUTER_MODEL")     # from .env (optional)
            or "qwen/qwen2.5-14b-instruct"       # default free model
        )

        self.base_url = base_url

        # Load API key securely
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")   # loaded from .env
        )

        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY in environment (.env).")

        # default request headers
        self.default_headers = default_headers or {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://data-scientist-agentic-ai",
            "X-Title": "Data Scientist Agentic AI",
        }

    def chat_completion(
        self,
        prompt: str,
        role: str = "user",
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform an OpenRouter chat completion request.
        """

        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if extra_messages:
            messages.extend(extra_messages)

        messages.append({"role": role, "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(kwargs)

        logger.debug("OpenRouter Request Payload: %s", payload)

        response = requests.post(
            self.base_url,
            headers=self.default_headers,
            json=payload,
            timeout=60,
        )

        try:
            response.raise_for_status()
        except Exception as e:
            logger.error(
                "OpenRouter API Error: %s | HTTP Response: %s",
                e, response.text
            )
            raise

        data = response.json()
        logger.debug("OpenRouter Response: %s", data)
        return data
