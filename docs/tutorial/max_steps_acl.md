# Tutorial: MaxSteps ACL-V3 Curriculum Training

本教程说明如何在 AgentEvolver 中使用 **max_steps 的自动课程学习**（ACL V3 风格）。

> 重要：该功能 **只控制** `actor_rollout_ref.rollout.multi_turn.max_steps`（即 multi-turn 最大轮数），
> 不控制 `max_response_length / token budget / max_model_len`。


## 1. 功能概览

- 控制变量：`curriculum/current_max_steps`（float，内部状态）
- rollout 实际使用：`applied_max_steps = int(round(current_max_steps))`
- 目标计算：
  - 从 **成功 episode** 的 turns 分布估计能力边界：`P90(success_turns)`
  - `target = P90 + headroom_steps`
  - `current <- (1-α)*current + α*target`
  - clip 到 `[min_max_steps, max_max_steps]`

数据来源口径：
- turns：`len(traj.steps)`
- success：`traj.reward.outcome >= curriculum.success_outcome_threshold`
- clipped_by_max_steps：当前工程缺少显式 done_reason，因此用近似：
  - `turns >= applied_max_steps and (not traj.is_terminated)`


## 2. 你将获得的 SwanLab 指标

控制器层（rolling mean）：
- `curriculum/avg_exceed_rate`：rolling mean 的 `turns clip_ratio`
- `curriculum/avg_success_rate`
- `curriculum/current_max_steps`
- `curriculum/success_samples_count`

batch 层（即时统计）：
- `curriculum/success_rate`
- `curriculum/turns/clip_ratio`
- `curriculum/turns/min|max|mean|p50|p90`


## 3. 配置与脚本

我们提供了一个最小可用的训练配置与脚本（基于 `examples/basic.yaml` 风格）：

- `examples/max_steps_acl.yaml`
- `examples/run_max_steps_acl.sh`

### 3.1 修改/确认环境服务

例如 AppWorld：

```bash
bash env_service/launch_script/appworld.sh
```

然后确认 env_url 对应：默认脚本使用 `http://localhost:8080`。

### 3.2 设置 SwanLab（保留 swanlab-api-key 模式）

```bash
export SWANLAB_API_KEY=xxxxx
export SWANLAB_MODE=cloud          # 或 local
export SWANLAB_LOG_DIR=swanlog
```

### 3.3 启动训练

```bash
bash examples/run_max_steps_acl.sh
```

训练将：
- 使用 `examples/max_steps_acl.yaml` 作为主配置（仍然通过 hydra 引入 `ppo_trainer + agentevolver` 默认）
- 开启 `curriculum.enable_max_steps_control=true`
- 通过 `trainer.logger=['swanlab','console']` 输出到 SwanLab


## 4. 如何调参（建议）

- 初期建议：
  - `min_max_steps=10, max_max_steps=60`
  - `headroom_steps=2`
  - `smoothing_alpha=0.2`
- 如果 max_steps 变化太快：降低 `smoothing_alpha`
- 如果长期卡在较小 max_steps：降低 `min_success_samples_for_update_turns` 或调低 `success_outcome_threshold`


## 5. 常见问题

### Q1: 为什么 current_max_steps 不更新？
成功样本不足：`success_samples_count < min_success_samples_for_update_turns`。

### Q2: applied_max_steps 如何核对？
每条 trajectory 会写入：`trajectory.metadata['applied_max_steps']`，并且训练 metrics 里也会记录 `curriculum/applied_max_steps`（用于 debug 对齐）。

