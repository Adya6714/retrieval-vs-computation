"""
OpenRouter client covering GPT-4o, o3, and other 
closed models behind one key. Dormant until OPENROUTER_API_KEY is set in .env.
Interface identical to MockClient.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenRouterClient:
    def __init__(self, model: str = "openai/gpt-4o") -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Add it to .env before running the sweep.")
        
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True
    )
    def _make_api_call(self, prompt: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        if not response.ok:
            try:
                body = response.json()
                detail = body.get("error", {}).get("message") or str(body)[:400]
            except Exception:
                detail = (response.text or "")[:400]
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} for {response.url}: {detail}",
                response=response,
            )
        return response.json()

    def complete(self, problem_id: str, prompt: str, **kwargs: Any) -> dict:
        try:
            data = self._make_api_call(prompt)
            
            choices = data.get("choices", [])
            response_text = ""
            if choices and isinstance(choices, list):
                response_text = choices[0].get("message", {}).get("content", "")
                
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return {
                "response": response_text,
                "model": self.model,
                "problem_id": problem_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        except Exception as e:
            return {
                "response": f"ERROR: {str(e)}",
                "model": self.model,
                "problem_id": problem_id,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

    def complete_batch(self, problems: list[dict], **kwargs: Any) -> list[dict]:
        # TODO: replace with async batching when throughput matters.
        results = []
        for prob in problems:
            pid = prob.get("problem_id", "unknown")
            prompt = prob.get("prompt", "")
            results.append(self.complete(pid, prompt, **kwargs))
        return results
