# HP-25 W7 language transform (Hindi first)
addresses: [[D05_Cross_Linguistic]] · phase: Track T · needs: Colab T4 (open models) or small API budget (closed models); translation review

PROMPT:
Goal: add a translation surface transform and test whether it behaves like W1 (paraphrase) or W3 (isomorph/rename) once entities are or aren't localized.
Steps:
1. Select 30 GSM canonical-correct items (Acc_can ≥ .30, at least 2 models).
2. Produce two Hindi renderings per item: (a) entities kept in English/transliterated, numbers and structure unchanged; (b) entities localized to Hindi-appropriate names, numbers and structure unchanged. Gold fixed in both (translation is meaning-preserving).
3. Back-translate both renderings to English and confirm semantic equivalence to canonical before any model call; log any item that fails this check and exclude it.
4. Run the existing GSM verifier on gold before any model call.
5. Run on the [[D13_Continued_Pretraining_Transfer]] models (T4) and, if API budget allows, the original 5-model roster.
Output: `data/problems/w7_hindi_pilot.csv` (new small bank, separate from the main GSM bank), `results/raw/T8_W7_hindi.csv`, `results/derived/T8_W7_metrics.csv`, `docs/trackT/T8_REPORT.md`.
Validate: back-translation check passes for every item used; report exclusions.
