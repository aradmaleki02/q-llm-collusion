# Algorithmic Diversity Can Fail: Q–LLM Collusion

A tolerant Q-learning pricing agent learns to sustain collusion with three LLM rivals and gains a small late-game profit advantage over its partners. Ordinary Q-learning improves profit but does not sustain collusion. The contrast challenges treating algorithmic diversity alone as a safeguard against AI price collusion.

This repository contains the study implementation for the **Apart Research AI Collusion Research Sprint, October 2026**. External server outcome logs and measured figures are not included in this code release.

## Study setup

| Setting | Value |
|---|---|
| Market | Four sellers, simultaneous prices, logit demand, own-profit rewards |
| Comparisons | Ordinary Q + three LLMs; tolerant Q + three LLMs; four-LLM reference |
| Server training | 120 games per learner per seed; 3,000 rounds per game; seeds 0, 1, 2 |
| Exposure | 360,000 fresh transitions per learner per seed |
| LLM | `gpt-6-luna`, low reasoning, complete current-game history, private notes up to 160 words |
| Q network | 128-unit GRU, latest-observation skip connection, 128-unit ReLU head |
| Learning | Double DQN, Adam `3e-4`, smooth-L1 loss, discount `0.95` |
| Ordinary / tolerant Q | Negative-TD loss weight `1.0` / `0.25` (`optimistic` in the CLI) |
| Replay | 5,000 transitions; batch 8; four updates per fresh transition after 64-transition warmup |
| Exploration | `0.20` → `0.05` over the first 1,000 total training steps; then `0.05` |
| Evaluation | Fresh games; greedy Q actions; learning and exploration disabled |

Weights, optimizer state, and recent replay persist across training games. Market history and private LLM notes reset each game. No seller observes another seller's current choice or private notes. With discount `0.95`, the stopping round is unannounced and the Q target bootstraps through the administrative cutoff.

## Setup

Python 3.11+ and PyTorch 2.5+ (below 3) are required. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`config.json` uses `cuda:0`; select `cpu` if needed. For live calls, copy `.env.example` to `.env` and enter your API key locally. The key file is ignored by Git.

## Run

Offline smoke run, with simulated opponents and no API calls:

```powershell
python -m collusion run --name smoke --backend mock --rounds 12 --train-episodes 1 --eval-episodes 1
```

Server-study configuration:

```powershell
python -m collusion run --name server-study --rounds 3000 --train-episodes 120 --seeds 0 1 2 --variants ordinary optimistic --eval-episodes 1 --budget 50
```

`--budget` is a shared lifetime spending cap, including prior charges and outstanding reservations, not an estimate of the full study's cost. Choose the intended cap before starting; a run can pause before finishing. The command above does not execute until you run it.

Resume or export reports:

```powershell
python -m collusion resume --name server-study
python -m collusion report --name server-study
```

The fixed-suite runner trains 120 games **for each variant and each seed**, then evaluates each frozen learner and a four-LLM reference. Checkpoints, the request ledger, and raw rounds are saved in `runs/`; CSV metrics and charts are exported to `reports/`. Both directories are excluded from this repository.

## Measures

- **Own profit:** total and discounted profit for seller 4; compare with the corresponding seller in the four-LLM reference.
- **Sustained collusion:** a 100-round window with cross-seller price dispersion at most 5% every round, mean price at least 5% above the competitive benchmark, and normalized market profit index at least 0.10.
- **Asymmetric gains:** Q profit minus mean contemporaneous LLM-rival profit; tail summaries use the final 100 rounds.

Report independent seeds separately. Higher own profit alone does not establish sustained collusion. The main novelty is cross-algorithm cooperation and unequal gains; Q-versus-Q collusion is already established in prior work.

## Code and tests

| File | Purpose |
|---|---|
| `collusion/market.py` | Demand, profit, competitive and joint-profit benchmarks |
| `collusion/learner.py` | Recurrent ordinary/tolerant Q-learning |
| `collusion/engine.py` | Fixed training/evaluation suite |
| `collusion/api.py` | LLM prompts, full-history caching, response validation |
| `collusion/storage.py` | Checkpoints, request identity, shared budget ledger |
| `collusion/report.py` | Per-seller metrics and CSV/HTML export |

```powershell
python -m unittest discover -s tests -q
```

Tests use mocked API calls. The existing `study_controller.py` and `study_protocol.json` retain the local adaptive-study workflow, which requires its saved baseline and initial policies. That separate controller adds 512 replay updates per completed game; the fixed-suite command above does not. `local_pretrain.py` and `capacity_audit.py` support local controls.

## Related work and provenance

Keppo et al. found that Q–LLM heterogeneity disrupted collusion in their tested duopoly. This project examines a different, four-seller setting and tolerant learning as a failure mode for treating diversity as a sufficient safeguard.

- [Keppo et al., *On the Fragility of AI Agent Collusion* (2026)](https://arxiv.org/abs/2603.20281)
- [Calvano et al., *Artificial Intelligence, Algorithmic Pricing, and Collusion* (2020)](https://doi.org/10.1257/aer.20190623)
- [Matignon et al., *Hysteretic Q-Learning* (2007)](https://gjlaurent.github.io/assets/papers/Matignon2007hysteretic.pdf)

Project author: **Arad Maleki**. OpenAI Codex assisted with implementation, analysis, and writing. The author ran the external server study and supplied its behavioral findings and training setup. This release contains the supplied implementation with a 3,000-round default and a focused README; it does not include credentials, model checkpoints, private notes, or fabricated outcome data.
