# D09 — Can Rename-Robustness Be Trained In? (intervention study)
status: deferred until after D1-lite

**Question.** Fine-tune on renamed/graded-distance variants of family A; does W3 robustness transfer to family B (same model)? If yes, fragility is a trainable surface-binding skill → direct practitioner guidance ("augment with entity-renamed data"). If no, fragility is family-specific memorization → augmentation snake oil.

**Why deferred.** It presupposes a validated instrument to measure the outcome; running it before D1-lite risks interpreting noise. Also partially anticipated by the K&K finding that fine-tuning improves generalization despite memorization ([[P17_Xie_KK_Memorization_2024]]) — the transfer-across-families question is the open part.

**Sketch.** LoRA on Llama-3.1-8B; train on GSM W3-ladders; evaluate ALGO/BW W3 + Commitment Depth shift. One GPU-day. Natural companion to D1-lite (same infra, opposite direction of inference).
