# GPU runbook — mechanistic Llama (Step 21)

ssh adya_srivastava_2023@172.24.16.177

AdyaSrivastava@2023

VPN alone does **not** give you a GPU. FortiClient only puts you on the lab/campus network.
You still need an **SSH login host** (and usually a GPU node after that).

## 0. What you need from whoever runs the GPUs

Ask lab / IT / the person who gave you VPN access for:

1. **SSH host** — hostname or IP (e.g. `login.xxx.edu` or `10.x.x.x`)
2. **Username**
3. Whether you use a **login node → GPU node** hop (Slurm/`srun`, or a second SSH)
4. Whether your **public key** is already authorized (`~/.ssh/id_ed25519.pub`)

Put those in `secrets/gpu.local.env` (gitignored). Never commit real hosts/credentials.

```bash
cp secrets/gpu.local.env.example secrets/gpu.local.env
# edit secrets/gpu.local.env
source secrets/gpu.local.env
```

## 1. Connect from your Mac terminal

With VPN connected (FortiClient shows Connected):

```bash
source secrets/gpu.local.env
ssh -p "${GPU_PORT:-22}" "${GPU_IDENTITY:+-i $GPU_IDENTITY}" "$GPU_SSH"
```

First-time checks if SSH hangs:

```bash
# Can you resolve / reach the host at all?
ping -c 2 "$GPU_HOST"          # ICMP may be blocked; don't panic if ping fails
nc -vz -w 5 "$GPU_HOST" "${GPU_PORT:-22}"   # can you open TCP 22?
```

If `nc` fails: VPN connected to the wrong profile, wrong host, or SSH not allowed from your subnet — go back to IT.

## 2. Once on the machine — confirm a GPU

```bash
nvidia-smi                  # should list GPU(s)
which python3
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If `nvidia-smi` works on a **login** node but `torch.cuda` is false, you may need to request a GPU interactive job (common on clusters):

```bash
# examples — use whatever your site documents
srun --gres=gpu:1 --pty bash
# or:  salloc --gres=gpu:1 && srun --pty bash
```

## 3a. Appendix H follow-up — Llama GSM + SP (priority)

Corrected Llama-3.1-8B-Instruct probe (content-gold, chat-direct only, greedy
behavior labels on the same backbone). Scope: GSM + ALGO shortest-path only
(no WIS/CC/BW). Expected instances ≈ 44+44+24 GSM + 55+55+50 SP ≈ 272.

```bash
# on GPU box (Colab T4 ok with bf16; resume-safe)
export HF_TOKEN=...   # gated Llama
tmux new -s mech_llama_gsm_sp
bash scripts/runs/launch_mechanistic_llama_gsm_sp.sh gsm   # run GSM first
bash scripts/runs/launch_mechanistic_llama_gsm_sp.sh sp    # then SP
# or: bash scripts/runs/launch_mechanistic_llama_gsm_sp.sh all
```

Outputs:

| File                                                      | Role                                             |
| --------------------------------------------------------- | ------------------------------------------------ |
| `results/raw/mechanistic_llama_gsm_sp_raw.csv`            | long form: one row per (problem, variant, layer) |
| `results/derived/mechanistic_llama_gsm_sp_summary.csv`    | Wilcoxon can vs W3/W6                            |
| `results/derived/mechanistic_llama_behavior_link.csv`     | Mann-Whitney correct vs incorrect                |
| `results/derived/mechanistic_llama_gold_distribution.csv` | gold / first-token frequency audit               |

Dry-run (no GPU): `python3 scripts/run_mechanistic_llama_gsm_sp.py --dry-run --limit 2`

## 3b. Content-gold mechanistic sweeps (same banks)

`--gold-token-mode content` ranks the first **content** gold token (SP node
after `Path:`, CC digit after `Count:`, WIS first `Selected` value), not format
scaffolding. Three runs separate model (Instruct vs base) from prompt mode
(chat-direct vs raw-qa):

```bash
# on GPU box, after syncing repo + deps (transformer-lens, torch+CUDA, accelerate)
export HF_TOKEN=...   # needed for gated Llama

tmux new -s mech_contentgold
bash scripts/runs/launch_mechanistic_contentgold.sh all
# or one at a time: qwen-instr | llama | qwen-base
```

Outputs (do not overwrite legacy scaffold-gold CSVs):

| Run                  | Model                              | Prompt      | Output                                                                         |
| -------------------- | ---------------------------------- | ----------- | ------------------------------------------------------------------------------ |
| Instruct chat-direct | `Qwen/Qwen2.5-7B-Instruct`         | chat-direct | `results/raw/mechanistic_sweep_qwen25_7b_instruct_chatdirect_contentgold.csv`  |
| Instruct chat-direct | `meta-llama/Llama-3.1-8B-Instruct` | chat-direct | `results/raw/mechanistic_sweep_llama31_8b_instruct_chatdirect_contentgold.csv` |
| Base raw-qa          | `Qwen/Qwen2.5-7B`                  | raw-qa      | `results/raw/mechanistic_sweep_qwen25_7b_base_rawqa_contentgold.csv`           |

Expected n: ALGO can/W6 ≈100/90, BW 65/65, GSM 44/24 (~398 rows each).

After the three sweeps the same launcher runs **Llama ALGO forced-greedy**
(`scripts/algo_llama_greedy_accuracy.py`, `do_sample=False`) then the
**pass/fail gate** (`scripts/runs/mechanistic_contentgold_gate.py`):

- **FAIL** if greedy ≈6% but median content-gold final rank ≈1 (or format keywords still targeted).
- **PASS** if greedy ≈6% and median content-gold final rank is high (thousands).

Paper Llama ALGO SP cells (.059 / .048): see
`results/derived/LLAMA_ALGO_CANONICAL_PROVENANCE.md` (frozen 2/34 and 1/21;
OpenRouter 7/111 was not forced-greedy).

## 4. Optional SSH config (local only)

Add to **your** `~/.ssh/config` (not this repo):

```
Host rvc-gpu
  HostName FILL_IN
  User FILL_IN
  Port 22
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

Then: `ssh rvc-gpu`
