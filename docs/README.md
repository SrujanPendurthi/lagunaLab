# Documentation index

Start with the top-level [README](../README.md) for the summary and the one
command that runs the 16 GB configuration. Then, by question:

| If you want to know… | Read |
|---|---|
| What the answer is for a 16 GB laptop, and what is unmeasured | [part3-answer.md](part3-answer.md) |
| How the code is laid out and how data moves through it | [architecture.md](architecture.md) |
| What was done, in what order, with which commands, and why | [history.md](history.md) |
| Every finding, with interpretation and caveats | [results.md](results.md) |
| Which file holds which number, and every run's raw table | [results-catalog.md](results-catalog.md) |
| What was tried and dropped, with numbers | [no-gos.md](no-gos.md) |
| How to build models, serve, reproduce results and stay safe on a 24 GB Mac | [runbook.md](runbook.md) |
| What moeStream relies on, and what is safe to change | [interfaces.md](interfaces.md) |
| How another repo reuses the replay harness, references and gates | [parity-harness.md](parity-harness.md) |
| What is still open, unmeasured or limited | [open-work.md](open-work.md) |
| The design for removing the per-layer sync, and why stage 1 failed | [gpu-side-routing.md](gpu-side-routing.md) |
| The original specs and task-by-task plans | `superpowers/specs/`, `superpowers/plans/` |

Conventions used throughout:

- Every number names the `results/` file it comes from. Speeds are comparable
  only within one file.
- "Peak" is `mx.get_peak_memory()` (excludes MLX's free-buffer pool);
  "footprint" is the process footprint from the watchdog or `top`.
- "Resident" means all weights in memory; "streaming" means routed experts
  read from the SSD into a fixed number of slots per layer.
- "Pruned-50" is REAP, uniform, 50% of routed experts removed (128 of 256
  kept per layer).
- "`laptop16`" is pruned-50, streamed, 64 slots, 1 GB MLX pool cap, plain KV.
- Hardware for everything: 24 GB M4 MacBook Air, MLX 0.32.0, mlx-lm 0.31.3.
