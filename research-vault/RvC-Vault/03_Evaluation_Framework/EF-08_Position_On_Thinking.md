# EF-08 Position note: what this programme says about "can LLMs think"

status: position (reasoning note, no new numbers) · bears_on: CC-1, CC-2, [[EF-07_W3_Construct_Audit]], [[BX-02_Transfer_and_Concept_Tests]], [[BX-03_Memory_Science_Probes]]
use: Paper I discussion paragraph; site section "What this measures, and what it does not"; talks.

## The argument we are responding to
"LLMs cannot think" is unscientific without a rigorous definition of thinking and an argument that LLMs reach human-like outputs through non-equivalent causal structure. Vision models were shown to share causal structure with primate vision, so non-equivalence would be surprising, and LLMs are not inferior to human cognition.

## Where we agree
- A model-level yes/no about "thinking" is not a scientific claim without an operational definition and a causal comparison.
- Output-level failures alone do not establish non-thinking.

## Where the argument overreaches (quarantined external claims; verify before citing)
- **Vision equivalence.** The evidence (Yamins and DiCarlo line; Brain-Score) is that deep networks predict ventral-stream responses better than earlier models: partial representational similarity, not causal equivalence. The same networks are texture-biased where humans are shape-biased (Geirhos et al. 2019) and fail on perturbations humans ignore.
- **"Not inferior."** Our own data contradicts a blanket claim: Claude .172 on canonical Blocksworld; 14 obfuscated planning items fail every model (F4).

## What our measurements actually support
- **Operational stance.** We do not ask whether a model thinks. We ask, per instance, whether an answer was produced by a procedure invariant to answer-irrelevant surface features. That is graded and measurable.
- **Numeric regeneration is nearly free (F1, W6).** The procedure generalises over values within a template. That rules out pure item lookup for those items.
- **Cover-story transfer is expensive (F1, W3 as built = isomorph, see EF-07).** Humans show the same pattern: problem isomorphs with identical structure differ several-fold in difficulty by cover story (Kotovsky, Hayes and Simon 1985), and analogical transfer across cover stories is poor without cues (Gick and Holyoak 1980). So W3 cost is not evidence of non-thinking; it is a point of behavioural similarity whose mechanism is open.
- **Register sensitivity (F2) and direction effects (F3)** also have human analogues (content effects in the Wason task; forward vs backward search). Again not evidence either way on its own.
- **Fragility is model × item (F4) and robustness dissociates from accuracy (F5).** "Does model M think?" is ill-posed at model level. The answer varies by instance.
- **Mixture IRT selects K=1 on arithmetic (F7).** No discrete strategy classes at this n. If this holds at scale, "retrieval vs computation" is a continuum, which undercuts both "they only retrieve" and "they reason like us".
- **Source monitoring (CC-2, gated).** Humans are also poor at source monitoring (Johnson, Hashtroudi and Lindsay 1993). If models fail it too, that is a similarity. This is exactly the causal-structure comparison the argument asks for, which is why CC-2 is in the plan.

## One-paragraph version for papers
We do not ask whether language models think. We measure, per problem instance, whether a correct answer was produced by a procedure that is invariant to answer-irrelevant surface features. Across five primary models and three task families this property is graded, model-item specific, and dissociable from accuracy, and several of its failure patterns (cover-story isomorphs, representational register, search direction) have direct analogues in human problem solving. The question our instrument makes empirical is therefore not "thinking or not" but which surface features a given solution procedure is bound to, and whether that binding resembles the human one.

## Discipline
- No new numbers here. Every number cites a finding ID.
- External citations above are quarantined until full-text notes exist in 05_Papers ([[P-00_Ingestion_Queue]]).
