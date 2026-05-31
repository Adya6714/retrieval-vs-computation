"""
Mock client for local pipeline testing. Drop-in replacement for real API clients. 
Pass response_map to control per-problem responses in tests.
"""

from __future__ import annotations


class MockClient:
    def __init__(self, response_map: dict | None = None, default_response: str = "The answer is 42."):
        self.response_map = response_map if response_map is not None else {}
        self.default_response = str(default_response)

    def complete(self, problem_id: str, prompt: str, **kwargs) -> dict:
        try:
            response_text = self.response_map.get(problem_id, self.default_response)
        except Exception:
            response_text = self.default_response

        try:
            prompt_tokens = len(str(prompt).split())
        except Exception:
            prompt_tokens = 0

        return {
            "response": response_text,
            "model": "mock",
            "problem_id": problem_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 10
        }

    def complete_batch(self, problems: list[dict], **kwargs) -> list[dict]:
        results = []
        for prob in problems:
            try:
                pid = str(prob.get("problem_id", ""))
                prompt = str(prob.get("prompt", ""))
                results.append(self.complete(pid, prompt, **kwargs))
            except Exception:
                # In worst case scenario, append a safe fallback dict to keep length identical
                results.append({
                    "response": self.default_response,
                    "model": "mock",
                    "problem_id": prob.get("problem_id", ""),
                    "prompt_tokens": 0,
                    "completion_tokens": 10
                })
        return results
