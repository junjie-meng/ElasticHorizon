# QuickStart

This guide provides two distinct paths for training an agent:

1. **Basic GRPO Training**: A standard method to get started quickly.
2. **AgentEvolver Training**: An advanced method that supports a self-evolving agent training.



## Prerequisites: One-Time Global Setup

Before you begin, run these commands in your terminal to configure your environment. You only need to do this once.

1. Initialize Conda

```bash
source <YOUR_CONDA_PATH>/etc/profile.d/conda.sh
```

2. Configure API Endpoints

```bash
export DASHSCOPE_API_KEY="<YOUR_API_KEY>"
export HF_ENDPOINT=https://hf-mirror.com
```

> 💡 **Tip:** Add the `export` commands to your `~/.bashrc` or `~/.zshrc` file to set them automatically in new terminal sessions.



## Part A: Basic GRPO Training

### Step 1: Setup Env-Service (AppWorld for example)


This launches the simulation environment (e.g., AppWorld) where the agent will operate. *This service will run in the background. You'll need a new terminal for the next step.*


```bash
conda activate appworld
bash env_service/launch_script/appworld.sh
```

### Step 2: Start Basic GRPO Training

This command starts the training process using the GRPO method.


```bash
conda activate agentevolver
bash examples/run_basic.sh
```



## Part B: AgentEvolver Training


### Step 1: Setup Env-Service (AppWorld for example)

Just like in basic training, this launches the agent's simulation environment. This service will run in the background. You'll need a new terminal for the next step.


```bash
conda activate appworld
bash env_service/launch_script/appworld.sh
```

### Step 2: Setup ReMe-Service

This service gives the agent long-term memory and the ability to reflect on past actions. *This service will listen for requests on http://127.0.0.1:8001. Keep this terminal open.*

Configure API Endpoints:

```bash
export FLOW_EMBEDDING_API_KEY="<YOUR_API_KEY>"
export FLOW_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export FLOW_LLM_API_KEY="<YOUR_API_KEY>"
export FLOW_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

```bash
conda activate reme
cd external/reme
reme \
  config=default \
  backend=http \
  thread_pool_max_workers=256 \
  http.host="127.0.0.1" \
  http.port=8001 \
  http.limit_concurrency=256 \
  llm.default.model_name=qwen-max-2025-01-25 \
  embedding_model.default.model_name=text-embedding-v4 \
  vector_store.default.backend=local \
  op.rerank_memory_op.params.enable_llm_rerank=false
```


### Step 3: Start AgentEvolver Training
With the environment and ReMe services running, start the AgentEvolver training.

```bash
conda activate agentevolver
bash examples/run_overall.sh
```
<<<<<<< HEAD
=======

------

## Part C: Multi-Node Training

> Please read Part A and Part B first.

To run AgentEvolver and train agents on multiple nodes, we need some extra setup.

### Step 0. Prepare the Training

Follow the instructions in [Configuration](./configuration.md) to prepare the training environment on **all** nodes.

### Step 1. Start the Ray cluster

In Part A and Part B, we leave the Ray cluster to be started by AgentEvolver. To enable multi-node training, we need to start the Ray cluster manually.

Start Ray on *all* nodes:

```bash
# all ray node must be in the same conda environment
conda activate agentevolver

# node 1 as head
ray start --head

# other nodes as followers
ray start --address='<head addr>'
```

### Step 2. Start Env-Service (AppWorld for example)

Start envservice on *one* node:

```bash
conda activate appworld
cd env_service
bash launch_script/appworld.sh
```

Once you see the successful startup logs, you're good to go. The address of this node will be used in the next step.

If ReMe-Service is needed, start it on *one* node.

### Step 3. Start AgentEvolver

Create a launch script based on [run_basic.sh](https://github.com/modelscope/AgentEvolver/blob/main/examples/run_basic.sh) or [run_overall.sh](https://github.com/modelscope/AgentEvolver/blob/main/examples/run_overall.sh):

- Update `env_url` to the address of the envservice node
- Adjust other training parameters as needed

Then start training from *one* node:

```bash
conda activate agentevolver
bash run_basic.sh
```

AgentEvolver will automatically use the Ray cluster and spread the training tasks to all nodes.
>>>>>>> origin/feature/max-steps-curriculum
