# ml-sleep

## Development Workflow
- **All ML code runs on beauty** (remote GPU server) via SSH (`ssh beauty`)
- Edit and run directly on beauty — do NOT do local edit/push/pull cycles
- Local clone is for reference and version control only
- Beauty working directory: `~/sleep/ml-sleep`
- Commit and push from beauty when ready

## beauty Server
- 2x NVIDIA RTX 4090
- Always check `nvidia-smi` before picking a GPU
- Use `CUDA_VISIBLE_DEVICES=N` to select GPU
- **NEVER kill another user's process** — always verify PID ownership before any kill command
- Python: `python3`, venvs per project subfolder

## Repo Structure
Each project/paper gets its own subfolder with independent dependencies:
```
ml-sleep/
├── gandalf/          # NPP sleep staging paper (ResNet-LSTM)
├── <next-project>/   # future work
└── CLAUDE.md
```

Each subfolder has:
- `pyproject.toml` — install with `pip install -e .` inside a venv
- `configs/` — YAML hyperparams and paths
- `scripts/` — CLI entry points for training/eval
- `src/` — importable package code

## Data
- Data lives on beauty at `~/sleep/` (e.g., `~/sleep/pt_ekyn/`, `~/sleep/pt_ekyn_robust_50hz/`)
- Data is NOT in the repo — `.gitignore` excludes it
- Config YAML files point to data paths on beauty

## Related Repos on beauty
- `~/sleep/sleep-stage-classification-tutorial` — tutorial repo
- `~/sleep/experiments/` — training outputs (models, plots, configs)
