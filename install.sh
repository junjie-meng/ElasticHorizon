#!/usr/bin/env bash
set -e

echo "Installing AgentEvolver environment..."
echo

# ---- Step 1. Check Conda installation ----
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not found in PATH."
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ---- Step 2. Ask user for environment name ----
ENV_NAME="agentevolver"
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "⚠️  Environment '$ENV_NAME' already exists. If you need to reinstall it, please delete the existing environment first."
    exit 1
fi

# ---- Step 3. Create new environment ----
echo
echo "📦 Creating environment '$ENV_NAME'..."
conda create -y -n "$ENV_NAME" python=3.11

# ---- Step 4. Activate environment ----
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- Step 5. Install CUDA toolkit ----
echo
echo "🚀 Installing CUDA toolkit ..."
# 锁定 CUDA Toolkit 版本为 12.4 以匹配 PyTorch 2.6.0+cu124
conda install -y -c nvidia cuda-toolkit=12.4

# ---- Step 6. Install Python dependencies ----
if [[ ! -f requirements.txt ]]; then
    echo "⚠️  No requirements.txt found in current directory. Please check your working directory."
    exit 1
else
    echo
    echo "📥 Installing packages from requirements.txt ..."
    # 配置阿里云镜像源以加速下载并解决依赖构建时的网络问题
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    pip config set global.trusted-host mirrors.aliyun.com

    pip install -r requirements.txt
fi

# ---- Step 7. Install FlashAttention packages ----
echo
echo "⚙️  Installing flash-attn libraries ..."
# 安装本地的 flash-attn wheel 包
echo "📦 Installing local flash-attn wheel..."
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 安装 ring-flash-attn
echo "⚙️  Installing ring-flash-attn..."
pip install ring-flash-attn

pip install agentscope==1.0.10
pip uninstall -y opentelemetry-exporter-prometheus

# ---- Step 8. Finish ----
echo
echo "✅ Installation complete!"
echo "Environment '$ENV_NAME' is ready for your AgentEvolver! Please follow the rest instructions to start training."
echo
echo "To activate it later, run:"
echo "  conda activate $ENV_NAME"
echo
