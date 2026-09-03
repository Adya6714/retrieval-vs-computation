# Idle Cells — Data Collected, Analysis Missing or Thin

Frame: **33 cells** = 3 families × (7 P1 variants + 2 P2 phase-groups [CCI / TEP] + 2 P3 arms [Infini-gram / mechanistic]).

A cell is **idle** when raw/rescored rows exist (n>0) but there is no dedicated derived analysis beyond bulk accuracy / CSS aggregation.

## Probe 1 (family × variant)

- ALGO × P1 × canonical: collected (valid_sum=550; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- **IDLE ALGO × P1 × W1**: data exists (collected_sum=550, valid_sum=549; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- **IDLE ALGO × P1 × W2**: data exists (collected_sum=550, valid_sum=539; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- ALGO × P1 × W3: collected (valid_sum=497; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- **IDLE ALGO × P1 × W4**: data exists (collected_sum=550, valid_sum=526; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- **IDLE ALGO × P1 × W5**: data exists (collected_sum=250, valid_sum=162; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- ALGO × P1 × W6: collected (valid_sum=320; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- BW × P1 × canonical: collected (valid_sum=500; models=['Claude', 'DeepSeek', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- **IDLE BW × P1 × W1**: data exists (collected_sum=522, valid_sum=480; models=['Claude', 'DeepSeek', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- **IDLE BW × P1 × W2**: data exists (collected_sum=522, valid_sum=486; models=['Claude', 'DeepSeek', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- BW × P1 × W3: collected (valid_sum=491; models=['Claude', 'DeepSeek', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- **IDLE BW × P1 × W4**: data exists (collected_sum=522, valid_sum=485; models=['Claude', 'DeepSeek', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- BW × P1 × W5: collected (valid_sum=462; models=['Claude', 'DeepSeek', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- BW × P1 × W6: collected (valid_sum=384; models=['Claude', 'DeepSeek', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- GSM × P1 × canonical: collected (valid_sum=172; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- **IDLE GSM × P1 × W1**: data exists (collected_sum=260, valid_sum=172; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- **IDLE GSM × P1 × W2**: data exists (collected_sum=260, valid_sum=172; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- GSM × P1 × W3: collected (valid_sum=172; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.

- **IDLE GSM × P1 × W4**: data exists (collected_sum=260, valid_sum=172; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- **IDLE GSM × P1 × W5**: data exists (collected_sum=260, valid_sum=172; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) but only enters CSS/VRI/omnibus tables — no dedicated analysis.

- GSM × P1 × W6: collected (valid_sum=72; models=['Claude', 'GPT-4o', 'Gemini', 'Llama', 'o4-mini']) — **analyzed**.


## Probe 2 (family × phase-group)

- ALGO × P2 × CCI_plan_execution: collected; CCI analyzed in N2 (Gemini CCI all NaN).

- **IDLE ALGO × P2 × TEP_injection**: injection traces exist (n_rows_sum=244) but no standardized TEP / recovery analysis analogous to GSM.

- **THIN BW × P2 × CCI_plan_execution**: raw files exist (n_sum=300) but abort-dominated; no usable per-model CCI/TEP claim (protocol finding only). Counted secondary (not in primary 12).

- **THIN BW × P2 × TEP_injection**: raw files exist (n_sum=536) but abort-dominated; no usable per-model CCI/TEP claim (protocol finding only). Counted secondary (not in primary 12).

- GSM × P2 × CCI_plan_execution: collected (cci=True, tep=True) — **analyzed**.

- GSM × P2 × TEP_injection: collected (cci=False, tep=True) — **analyzed**.


## Probe 3 (family × arm)

- ALGO × P3 × infinigram: scored (n=116) — **analyzed** (triangulation / M1 / N5).

- ALGO × P3 × mechanistic: frequency-controlled Llama+Qwen — **analyzed** (N3).

- BW × P3 × infinigram: scored (n=65) — **analyzed** (triangulation / M1 / N5).

- **THIN BW × P3 × mechanistic**: partial/legacy rows exist but no family-complete P1-linked analysis.

- GSM × P3 × infinigram: scored (n=44) — **analyzed** (triangulation / M1 / N5).

- **THIN GSM × P3 × mechanistic**: partial/legacy rows exist but no family-complete P1-linked analysis.


## Primary idle list (12)

1. `P1/ALGO/W1`
2. `P1/ALGO/W2`
3. `P1/ALGO/W4`
4. `P1/ALGO/W5`
5. `P1/BW/W1`
6. `P1/BW/W2`
7. `P1/BW/W4`
8. `P1/GSM/W1`
9. `P1/GSM/W2`
10. `P1/GSM/W4`
11. `P1/GSM/W5`
12. `P2/ALGO/TEP_injection`

**Primary idle count: 12**


## Secondary thin cells (data present, analysis unusable or incomplete)

- `P2/BW/CCI_plan_execution`
- `P2/BW/TEP_injection`
- `P3/BW/mechanistic`
- `P3/GSM/mechanistic`

### Cell arithmetic
- 33 = 3 families × (7 P1 variants + 2 P2 groups + 2 P3 arms)
- Primary idle 12 = 11 P1 light variants (W1/W2/W4×3 + ALGO&GSM W5) + ALGO P2-TEP
- Analyzed remainder includes can/W3/W6 headlines, BW W5 sign-flip, GSM+ALGO CCI, all Infini-gram arms, ALGO mechanistic
