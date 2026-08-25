#!/bin/bash
# Train with the Elastic Horizon (adaptive max_steps) controller on AppWorld.
#
# ---- Start the environment service first ----
#   conda activate appworld
#   bash env_service/launch_script/appworld.sh

set -e

PROJECT_DIR="$(pwd)"
EXTRA_CFG_DIR="$PROJECT_DIR/examples"

env_url=${ENV_URL:-"http://localhost:8085"}

current_time=$(date "+%Y%m%d_%H%M%S")
experiment_name=max_steps_acl
log_file="log_${experiment_name}_${current_time}.log"

# SwanLab logging (optional; set SWANLAB_API_KEY to log to the cloud)
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-""}
export SWANLAB_MODE=${SWANLAB_MODE:-"local"}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"swanlog"}

python3 -m agentevolver.main_ppo \
  --config-path="$EXTRA_CFG_DIR" \
  --config-name='max_steps_acl' \
  env_service.env_url=$env_url \
  env_service.env_type=appworld \
  trainer.project_name="elastic_horizon_appworld" \
  trainer.experiment_name=$experiment_name \
  trainer.logger="['console','swanlab']" \
  trainer.n_gpus_per_node=${N_GPUS:-8} \
  trainer.nnodes=1 \
  trainer.total_epochs=70 \
  trainer.save_freq=10000 \
  trainer.test_freq=10 \
  trainer.val_before_train=true \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_batch_size=32 \
  data.max_prompt_length=4000 \
  data.max_response_length=21580 \
  data.filter_overlong_prompts=true \
  data.truncation='error' \
  data.return_raw_chat=true \
  data.train_files=null \
  data.val_files=null \
  actor_rollout_ref.model.path=${MODEL_PATH:-"Qwen/Qwen2.5-14B-Instruct"} \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.9 \
  actor_rollout_ref.rollout.prompt_length=20480 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.max_model_len=25580 \
  actor_rollout_ref.rollout.use_qwen3=false \
  actor_rollout_ref.rollout.enable_request_id=false \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_steps=30 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.kl_loss_coef=0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  curriculum.enable_max_steps_control=true \
  curriculum.min_max_steps=5 \
  curriculum.max_max_steps=50 \
  curriculum.headroom_steps=3 \
  curriculum.smoothing_alpha=0.2 \
  curriculum.success_sample_window=100 \
  curriculum.metrics_window=50 \
  curriculum.min_success_samples_for_update_turns=10 \
  curriculum.success_outcome_threshold=0.5 \
  task_manager.n=0 \
  task_manager.mixture.synthetic_data_ratio=0.0 \
  task_manager.mixture.use_original_tasks=true \
  2>&1 | tee "$log_file"
