# DS-11 — Inverse Inference: From Fragility to Training-Data Structure
family: deeper theory (long game) · cost: high on open models

**Idea.** Run the arrow backwards. Instead of "known exposure → predicted fragility" (D1/D4), ask "observed fragility pattern → inferred exposure structure of an unknown model." On OLMo/Llama where influence functions or corpus counts are feasible, calibrate the fragility→exposure map; then apply it to closed models to make FALSIFIABLE claims about their training-data structure from behavior alone.

**Why this is a summit result.** It turns the instrument into a scientific instrument for studying models you cannot open — the way spectroscopy infers stellar composition from light. Even a partial, uncertainty-bounded version ("this closed model's WIS behavior is consistent with near-zero exposure to interval-scheduling problems") is the kind of claim that defines a subfield.

**Prerequisites.** DS-01 + D1 + D4 all landed and cross-validated; influence-function tooling (Ruis-style) on at least one open model. This is a Year-2 direction; noted now so the earlier work is designed to enable it.
