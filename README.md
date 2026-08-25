# Elastic Horizon

**Discovering the Effective Interaction Frontier in Agentic Reinforcement Learning**

> 📢 Accepted to **EMNLP 2026**.

## Overview

Existing multi-turn agentic RL methods rely on **open-loop, monotonically increasing** horizon schedules (linear or multiplicative curricula) that blindly scale the interaction budget toward a manually specified maximum, with no mechanism to detect **when scaling should stop**.

We propose the **Effective Interaction Frontier Hypothesis**: for a given policy and task distribution, there exists a dynamic boundary *H\** beyond which additional interaction steps yield diminishing returns while incurring linear growth in compute cost.

**Elastic Horizon** is a closed-loop horizon controller that probes the agent's competence boundary with the **P90 of successful trajectory lengths**, and adapts the interaction budget on the fly:

```
1. Collect rollouts under the current horizon budget K_t
2. Update the success buffer with successful trajectories
3. Estimate the competence boundary: B_t = P90(success_lengths)
4. Set the target: K_target = B_t + Δ (headroom)
5. Smooth update: K_{t+1} = (1-α) · K_t + α · clip(K_target, K_min, K_max)
```

It expands automatically when the agent demonstrates capability growth and contracts when approaching the task's complexity ceiling, cutting cumulative trajectory tokens by ~40% at comparable or better performance.

## Installation

```bash
conda create -y -n elastic_horizon python=3.11
conda activate elastic_horizon
conda install -y -c nvidia cuda-toolkit=12.4
pip install -r requirements.txt
```

See `install.sh` for the full setup script (FlashAttention etc. may be needed depending on your hardware).

## Quick Start

**1. Set up an environment service** (downloads benchmark data on first run):

```bash
bash env_service/environments/appworld/setup.sh   # AppWorld
bash env_service/environments/bfcl/setup.sh       # BFCL v3
```

**2. Launch training:**

```bash
python launcher.py --conf examples/max_steps_acl.yaml --with-appworld   # AppWorld
python launcher.py --conf examples/bfcl_acl.yaml --with-bfcl            # BFCL v3
```

Or use `examples/run_max_steps_acl.sh` to invoke the trainer directly against an already-running environment service.

### Configuration

Elastic Horizon is configured via the `curriculum` section (see `examples/*.yaml`):

| Key | Default | Description |
|-----|---------|-------------|
| `enable_max_steps_control` | `true` | Turn the controller on/off |
| `update_mode` | `acl` | `acl` = Elastic Horizon (ours); `scaling` / `tti` = open-loop schedule baselines |
| `min_max_steps` / `max_max_steps` | 5 / 50 | Hard bounds K_min / K_max |
| `headroom_steps` | 3 | Exploration headroom Δ |
| `smoothing_alpha` | 0.2 | EMA smoothing coefficient α |
| `success_sample_window` | 100 | Success buffer capacity |

The initial horizon K_0 is taken from `actor_rollout_ref.rollout.multi_turn.max_steps`.

### Code

- Controller: [`agentevolver/utils/max_steps_curriculum.py`](agentevolver/utils/max_steps_curriculum.py)
- Training-loop integration: [`agentevolver/module/trainer/ae_ray_trainer.py`](agentevolver/module/trainer/ae_ray_trainer.py)

## Experiments

We evaluate on **AppWorld** (complex multi-step API calling) and **BFCL v3** (function calling), training Qwen2.5 models with GRPO (KL coefficient = 0), against fixed-*K* horizons (K ∈ {10, ..., 50}), a linear schedule (ScalingInter-RL style), and an exponential schedule (TTI style).

## Acknowledgements

**Our work is based on [Agent Evolver](https://github.com/modelscope/AgentEvolver)** — we build the Elastic Horizon controller on top of its agentic RL training framework (which in turn builds on [verl](https://github.com/volcengine/verl)). We thank the authors for open-sourcing it.

## Citation

```bibtex
@inproceedings{elastichorizon2026,
  title     = {Elastic Horizon: Discovering the Effective Interaction Frontier in Agentic Reinforcement Learning},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
