"""
Anthropic API client. Dormant until ANTHROPIC_API_KEY 
is set in .env. Interface is identical to MockClient for drop-in substitution.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-5-20251001") -> None:
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env before running the sweep.")
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True
    )
    def _make_api_call(self, prompt: str) -> dict:
        headers = {
            "x-api-key": str(self.api_key),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    def complete(self, problem_id: str, prompt: str, **kwargs: Any) -> dict:
        try:
            data = self._make_api_call(prompt)
            
            content_blocks = data.get("content", [])
            response_text = ""
            if content_blocks and isinstance(content_blocks, list):
                response_text = content_blocks[0].get("text", "")
                
            usage = data.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

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
