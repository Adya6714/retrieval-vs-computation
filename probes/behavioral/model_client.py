import os
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PDDL_ROOT = "/Users/adya/Desktop/LLMs-Planning"


class ModelClient:
    """
    Single-turn API wrapper for OpenRouter.
    Every call to complete() is a completely fresh session.
    No conversation history ever accumulates.
    """

    def __init__(self, model_string: str, temperature: float = 0.0):
        self.model = model_string
        self.temperature = temperature
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable not set"
            )

    def complete(self, prompt: str) -> str:
        """
        Send a single user message. Returns the response string.
        Raises RuntimeError on non-200 status.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        resp = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=90
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"API error {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
