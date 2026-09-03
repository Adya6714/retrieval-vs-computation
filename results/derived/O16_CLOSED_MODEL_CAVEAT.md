# O16 closed-model caveat (paper)

Corpus ground-truth membership (exact / near-exact match in The Pile or Dolma
via Infini-gram) can be established for **Pythia** (Pile) and **OLMo** (Dolma)
because those pretraining corpora are open and indexed.

The same ground-truth cell **cannot** be constructed for Claude, GPT-4o, Gemini,
o4-mini, or DeepSeek: their pretraining data are proprietary. This is a
**permanent limitation** of contamination research on closed models—not a
temporary compute or API gap. Proxy scores (Infini-gram on RedPajama, LM
surprisal) remain available for those models but are not calibratable against
known membership.
