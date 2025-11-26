#!/bin/bash
# Installation script for Qwen-VL training environment
# For GPU server with CUDA 12.8

set -e  # Exit on error

echo "=========================================="
echo "Installing Qwen-VL Training Environment"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment 'qwen-vl'..."
conda create -n qwen-vl python=3.10 -y

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate qwen-vl

# Check CUDA version
echo ""
echo "Step 2: Detecting CUDA version..."
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "Using CUDA 12.8"
    CUDA_VERSION="cu128"
elif [ -d "/usr/local/cuda-12.2" ]; then
    export CUDA_HOME=/usr/local/cuda-12.2
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "Using CUDA 12.2"
    CUDA_VERSION="cu122"
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME=/usr/local/cuda-12
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "Using CUDA 12"
    CUDA_VERSION="cu121"
else
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "Using CUDA 11.8"
    CUDA_VERSION="cu118"
fi

nvcc --version

# Install PyTorch
echo ""
echo "Step 3: Installing PyTorch 2.5.1 (latest stable)..."
if [ "$CUDA_VERSION" = "cu128" ] || [ "$CUDA_VERSION" = "cu122" ]; then
    # For CUDA 12.x, use cu121 wheels
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" = "cu118" ]; then
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
fi

# Install transformers from source
echo ""
echo "Step 4: Installing transformers from source..."
pip install git+https://github.com/huggingface/transformers.git

# Install core training dependencies
echo ""
echo "Step 5: Installing training dependencies..."
pip install deepspeed==0.17.1
pip install accelerate==1.7.0
pip install peft==0.17.1

# Install flash-attention (this takes time)
echo ""
echo "Step 6: Installing flash-attention (this may take 5-10 minutes)..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install triton
echo ""
echo "Step 7: Installing triton..."
pip install triton==3.2.0

# Install qwen-vl-utils
echo ""
echo "Step 8: Installing qwen-vl-utils..."
cd ../qwen-vl-utils
pip install -e .
cd ../qwen-vl-finetune

# Install torchcodec for video support (optional)
echo ""
echo "Step 9: Installing torchcodec (optional, for video)..."
pip install torchcodec==0.2 || echo "Warning: torchcodec installation failed (optional)"

# Install additional useful packages
echo ""
echo "Step 10: Installing additional packages..."
pip install wandb tensorboard pillow

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "import flash_attn; print('Flash Attention: OK')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "    conda activate qwen-vl"
echo ""
echo "CUDA environment variables (add to ~/.bashrc):"
echo "    export CUDA_HOME=$CUDA_HOME"
echo "    export PATH=\$CUDA_HOME/bin:\$PATH"
echo "    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
