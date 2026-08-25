#!/usr/bin/env bash
set -e

echo "Installing ReMe environment..."
echo


# ---- Step 1. Ask user for environment name ----
ENV_NAME="reme"
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "⚠️  Environment '$ENV_NAME' already exists. If you need to reinstall it, please delete the existing environment first."
    exit 1
fi


# ---- Step 2. Create new environment ----
echo
echo "📦 Creating environment '$ENV_NAME'..."
conda create -y -n "$ENV_NAME" python=3.12


# ---- Step 3. Install ReMe package ----
echo "📦 Installing ReMe package..."
conda run -n "$ENV_NAME" pip install reme-ai

