# DS-09 — Compositional Stress and Length Generalization
family: new probe · cost: API-scale, existing generators

**Idea.** Your banks fix problem size. Vary structural complexity as a controlled axis: chain length, number of entities to bind, recursion depth, number of interacting constraints — at FIXED surface familiarity. Retrieval-consistent items should fail fast as composition grows (no stored answer for the bigger instance); computation-consistent items should degrade gracefully along a predictable curve. Cross this with W3 rename → the 2D map (surface distance × compositional depth) is a far richer fingerprint than either axis alone.

**Why it matters to the field.** Length/compositional generalization is a first-order open question (Dziri Faith-and-Fate; length-generalization literature). Framed as an axis of YOUR validated instrument — with intrusion, RIS, and Commitment Depth read at each depth — it becomes "at what compositional depth does retrieval stop substituting for computation," which is a mechanism question, not a benchmark.

**Prerequisite.** Family generators that parametrize size (BW block count, SP graph size, WIS interval count already do). Verifier re-check per depth.
