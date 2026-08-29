# BX-04 — Adaptation & Habituation (perceptual neuroscience's invariance probe)

## Origin
Neurons and infants both HABITUATE: repeat a stimulus and the response drops; present something the system treats as GENUINELY NEW and the response recovers (dishabituation). fMRI REPETITION SUPPRESSION uses exactly this: a brain region that treats two stimuli as "the same" shows reduced response on the second; if it treats them as different, no suppression. Infant looking-time studies (Baillargeon) use dishabituation to reveal what pre-verbal babies consider a new event. This is how you discover a system's INVARIANCE CLASSES without asking it anything.

## The deep principle
What a system considers "the same" is revealed by what it stops responding to. Adaptation is a direct read-out of the representation's equivalence classes. If a model's internal state "adapts" to a rename (treats renamed problem as the same), the representation is abstract; if it does not adapt (responds as if new), the representation is surface-bound.

## LLM translation (novel — almost no one does this)
- **Representational repetition suppression:** present canonical then W3 in sequence; measure change in internal activations / next-token distribution on the second relative to a genuinely-new problem. Small change = the model treats them as the same (abstract, robust); large change = treats rename as a new problem (surface-bound). A mechanistic invariance score that needs no correct answer. Complements RSA ([[DS-03_Representational_Invariance_RSA]]) with a dynamic, causal version.
- **In-context adaptation curves:** feed k surface-variants of the same structure in context; does accuracy/entropy stabilize (adaptation to structure) or stay volatile (no structural representation)? Adaptation RATE as a per-model signal.
- **Dishabituation test:** after adapting to a family, insert a structurally-novel-but-surface-similar item; a robust reasoner should "notice" (response recovers on the truly novel one, not on mere surface change).

## What it proves for the objective
Reveals the model's invariance classes directly — the cleanest possible operationalization of "does it see through the surface." Generates [[DS-17_Adaptation_Suppression_Probe]].

## Papers to be inspired by
- Repetition-suppression / fMRI adaptation literature (Grill-Spector) — the method source.
- Induction-heads and in-context-learning dynamics work (Olsson et al.) — the ML mechanism that would underlie in-context adaptation; connect but re-aim at invariance.
- Infant violation-of-expectation (Baillargeon, Onishi & Baillargeon 2005, cited in the false-belief results) — the behavioral logic.
