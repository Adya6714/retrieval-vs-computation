"""rvc11: corrected, single-path rerun pipeline for the retrieval-vs-computation study.

Design rules (enforced by ``python -m rvc11 qa``):
  * The bank stores an instruction-free task text and a formal spec per item.
  * Every prompt = task text + one programmatic answer-format block + one instruction block.
  * Scoring never reads prose. It checks the model's final-answer block against the formal spec.
  * Raw model calls are append-only JSONL. Scores are always re-derived from raw calls.
"""

__version__ = "1.1.0"
