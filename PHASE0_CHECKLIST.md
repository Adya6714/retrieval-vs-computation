# Phase 0 Checklist — Infrastructure and Housekeeping

Everything here is repo-and-env setup. Gate 0 is met when every box is checked.

---

## Step 1 — Create the GitHub repo

1. Go to github.com → New repository
2. Name: `retrieval-vs-computation`
3. Private (flip to public at submission)
4. Do NOT initialize with a README (we have our own)
5. Copy the remote URL

Then in your terminal:
```bash
cd wherever-you-keep-projects/
# Copy the repo folder we prepared here into your local machine, then:
git init
git remote add origin https://github.com/YOUR_USERNAME/retrieval-vs-computation.git
git add .
git commit -m "Phase 0: initial repo structure"
git push -u origin main
```

Add Shaswat and Nandini as collaborators: Settings → Collaborators → Add people.

---

## Step 2 — Python environment

```bash
conda create -n rvc python=3.11
conda activate rvc
pip install -r requirements.txt
```

Verify it worked:
```bash
python -c "import numpy, pandas, scipy, matplotlib, requests; print('OK')"
```

---

## Step 3 — API keys

1. Copy `.env.template` → `.env` (same folder, project root)
2. Fill in your keys:
   - OpenAI: platform.openai.com → API Keys
   - Anthropic: console.anthropic.com → API Keys
   - Infini-gram: no key needed
3. Run the test script:
```bash
python scripts/test_api_keys.py
```
All three checks should pass.

---

## Step 4 — Compute plan (research needed)

Answer these questions before Gate 0:

**A. BITS GPU cluster:**
- Does BITS Pilani give you student access to a GPU cluster?
- If yes: what GPU is available? (need ≥14GB VRAM for Qwen2.5-7B in fp16, or ≥8GB for 8-bit)
- Contact: your HPC admin or CS department

**B. Cloud GPU (if BITS cluster is unavailable or inadequate):**
- RunPod: runpod.io — cheapest, good for single-job runs
- Lambda Labs: lambdalabs.com — slightly pricier but more stable
- Vast.ai: vast.ai — cheapest of all, less reliable
- Colab Pro+: fine for testing, bad for long overnight runs
- Budget estimate: Qwen2.5-7B inference for ~300 problems × 6 variants = ~1800 forward passes → roughly 2-4 GPU hours on an A100 → ~$3-8

**Decision to record in CHARTER.md before Gate 0:**
- Compute source: [BITS cluster / RunPod / Lambda / other]
- GPU: [model]
- Estimated cost: [number]

---

## Step 5 — Overleaf

1. Go to overleaf.com → New Project → Upload Project
2. Download ACL 2024 template from: https://github.com/acl-org/acl-style-files
3. Name the project: `retrieval-vs-computation-paper`
4. Share with Shaswat and Nandini (Editor access)
5. Note the project URL in PROJECT_LOG.md

---

## Step 6 — Shared doc

Create one Google Doc or Notion page for the three of you:
- Paper outline (will fill in as work progresses)
- Meeting notes (use PROJECT_LOG.md format)
- TODO list

Link to it from PROJECT_LOG.md.

---

## Step 7 — Fill in CHARTER.md

Open `CHARTER.md` and complete:
- Division of Labor table (who owns what)
- Open decisions that are now resolved (compute, etc.)

Commit the updated charter:
```bash
git add CHARTER.md PROJECT_LOG.md
git commit -m "Phase 0: charter filled in, compute plan decided"
```

---

## Gate 0 Checklist

- [ ] GitHub repo created, Shaswat and Nandini added as collaborators
- [ ] Python env `rvc` created, `requirements.txt` installs without errors
- [ ] `python scripts/test_api_keys.py` passes all three checks
- [ ] Compute plan decided and recorded in CHARTER.md (BITS cluster or cloud GPU)
- [ ] Overleaf project created and shared
- [ ] Shared doc (Google Doc/Notion) created and linked
- [ ] CHARTER.md division-of-labor table filled in
- [ ] All files committed and pushed

**When all boxes are checked: tell me "Gate 0 passed" and we start Phase 1 (contamination triage).**
