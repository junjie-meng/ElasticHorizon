# Elastic Horizon: Adaptive MaxSteps Curriculum

## 背景与动机

Elastic Horizon 的核心动机是：**从成功样本（successful episodes）的 turns 分布估计“能力边界”**，并用 headroom + 平滑更新把训练难度逐步拉伸到模型当前能力边界附近。

在 AgentEvolver 中，多轮交互的上限由 `actor_rollout_ref.rollout.multi_turn.max_steps` 控制。若 max_steps 固定，可能出现：

- 过小：大量任务在接近成功时被提前截断（训练信号偏负、数据分布受限）
- 过大：大量无效长轨迹，吞吐下降、学习效率变差

本设计只实现 **max_steps 的自适应课程学习**（`current_max_steps`），不引入/不实现 token budget / max_response_length / max_model_len 的任何控制逻辑。


## 数据流 / 闭环流程

**每个 training step**（一次 rollout batch）执行如下闭环：

1. **rollout 前 (override)：**
   - controller 根据当前 `current_max_steps` 给出 `applied_max_steps = int(round(current_max_steps))`
   - 覆盖本次 rollout 使用的 `config.actor_rollout_ref.rollout.multi_turn.max_steps = applied_max_steps`
   - 将 `applied_max_steps` 写入每条 trajectory 的 `metadata['applied_max_steps']`（debug 对齐）

2. **rollout 生成：**
   - `ParallelEnvManager.rollout(...)` 产生 `List[Trajectory]`
   - `AgentFlow.execute(...)` 使用 `self.max_steps` 控制 loop：`for act_step in range(self.max_steps)`

3. **rollout 后 (统计)：**
   - 从每条 trajectory 抽取：
     - `turns = len(traj.steps)`
     - `success`：来自 `traj.reward.outcome`（通常 0/1），以 `outcome >= curriculum.success_outcome_threshold` 判定成功
     - `clipped_by_max_steps`：优先使用显式截断原因；当前工程缺省该字段，因此采用兜底近似：`turns >= applied_max_steps and (not traj.is_terminated)`
   - 聚合为 batch 指标：success_rate / clip_ratio / turns 的分位数

4. **update controller：**
   - controller 维护一个 successful turns 样本池（deque）
   - 估计能力边界：`p90 = P90(success_turns_pool)`
   - `target = p90 + headroom_steps`
   - `current = (1-α)*current + α*target`，并裁剪到 `[min_max_steps, max_max_steps]`
   - 下一次 rollout 前再次 override，从而**真实影响下一次多轮上限**


## 参数说明（Hydra config）

新增配置节点：`curriculum.*`

核心参数：

- `curriculum.enable_max_steps_control: bool`：是否启用 max_steps 课程学习
- `curriculum.min_max_steps: int`：current/applied 的下界
- `curriculum.max_max_steps: int`：current/applied 的上界
- `curriculum.headroom_steps: int`：目标值 headroom
- `curriculum.smoothing_alpha: float`：EMA 平滑系数 α

非核心但建议保留（默认值建议）：

- `curriculum.success_sample_window: int = 100`
- `curriculum.metrics_window: int = 50`
- `curriculum.min_success_samples_for_update_turns: int = 10`
- `curriculum.success_outcome_threshold: float = 0.5`


## 指标定义与口径（SwanLab / Tracking）

所有 key 必须以 `curriculum/` 开头，使用 `Tracking.log(data=..., step=global_step)` 上报。

### 控制器层（rolling mean）

- `curriculum/avg_exceed_rate`
  - **定义：rolling mean 的 turns clip_ratio**
  - 注意：不使用 token 越界，因为本任务不做长度控制
- `curriculum/avg_success_rate`
- `curriculum/current_max_steps`
- `curriculum/success_samples_count`

### batch 层（即时统计）

- `curriculum/success_rate`
- `curriculum/turns/clip_ratio`
- `curriculum/turns/max`
- `curriculum/turns/mean`
- `curriculum/turns/min`
- `curriculum/turns/p50`
- `curriculum/turns/p90`


## 与现有训练流程的集成点（最小侵入）

### 1) 初始化：创建 controller

文件：`agentevolver/module/trainer/ae_ray_trainer.py`

- 在 `fit()` 开始（Tracking 初始化后）创建 `MaxStepsCurriculumController`
- initial_current_max_steps 来自当前静态配置：`config.actor_rollout_ref.rollout.multi_turn.max_steps`

### 2) rollout 前：注入 max_steps_override

文件：`agentevolver/module/trainer/ae_ray_trainer.py`

- 在调用 `self.env_manager.rollout(...)` 之前：
  - `applied_max_steps = controller.get_max_steps_override()`
  - 临时覆盖 `self.config.actor_rollout_ref.rollout.multi_turn.max_steps = applied_max_steps`
  - 同时写 `metrics['curriculum/applied_max_steps'] = applied_max_steps` 便于对齐

### 3) rollout 后：统计 + update + 上报

文件：`agentevolver/module/trainer/ae_ray_trainer.py`

- 从 `trajectories` 抽取 turns/success/clipped，聚合 batch 指标
- `controller.update(success_rate=..., clip_ratio=..., successful_turns=[...])`
- 将 controller metrics + batch metrics 通过 `Tracking.log` 上报


## 边界情况与兜底策略

- **成功样本不足**：
  - 若 `len(success_turns_pool) < min_success_samples_for_update_turns`，controller 不更新 current_max_steps
  - reason 标记为 `insufficient_success_samples`

- **turns 缺失**：
  - turns 兜底为 `len(traj.steps)`；若 steps 为 None 则 turns=0 并跳过用于分位数计算

- **截断原因缺失**：
  - 当前 trajectory 没有 done_reason/truncated 字段
  - 采用近似：`turns >= applied_max_steps and (not traj.is_terminated)` 作为 clipped_by_max_steps

- **max_steps override 的生效保证**：
  - AgentFlow 在 init 时读取 config，因此需要在 rollout 前覆盖 config，并确保 env_manager 创建的 AgentFlow 读取到当次 config 值
  - 同时将 applied_max_steps 写入 trajectory.metadata 以便复盘对齐


指标（SwanLab）key 覆盖情况
你要求的这些已在训练 loop metrics 中产生，并最终通过 Tracking.log 统一上报（key 均以 curriculum/... 开头）：

控制器层（rolling mean）：
curriculum/avg_exceed_rate ✅（rolling mean clip_ratio）
curriculum/avg_success_rate ✅
curriculum/current_max_steps ✅
curriculum/success_samples_count ✅
batch 层：
curriculum/success_rate ✅
curriculum/turns/clip_ratio ✅
curriculum/turns/max ✅
curriculum/turns/mean ✅
curriculum/turns/min ✅
curriculum/turns/p50 ✅
curriculum/turns/p90 ✅