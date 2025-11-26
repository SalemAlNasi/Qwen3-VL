#!/bin/bash
#SBATCH --job-name=Qwen_Train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=t01pdscgpu04,t01pdscgpu35
#SBATCH --gres=gpu:8
#SBATCH --reservation=edge-ai-vla
#SBATCH --output=qwen_train_%j_%t.log
#SBATCH --error=qwen_train_%j_%t.err
#SBATCH --cpus-per-task=128

# (1) Load conda
source ~/miniconda3/bin/activate

# (2) Load environment
conda activate qwen-vl  # Qwen-VL environment

# (3) Set up distributed variables
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
EXP_SCRIPT="qwenvl/train/train_qwen.py"
GPUSPERNODE=8
DEEPSPEED_CONFIG="./scripts/zero3.json"

# SSL certificate fix for wandb (comment out if not needed)
# export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt  # Uncomment and adjust path if needed

echo "Master node: $MASTER_ADDR"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"

# (4) Choose config file and output directory from arguments
CONFIG="${1:-./scripts/slurm/configs/pointing_qwen25.yaml}"  # First argument or default
OUTPUT_DIR_ARG="${2:-}"  # Optional second argument for output directory

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

echo "Using config: $CONFIG"

# (5) Extract run name and wandb project from config
eval $(awk -F': ' '
NF==2 && $1 !~ /^#/ && $1 != "" {
    gsub(/^[ \t]+|[ \t]+$/, "", $1);
    gsub(/^[ \t]+|[ \t]+$/, "", $2);
    gsub(/"/, "", $2);
    if ($1 == "base_run_name") print "BASE_RUN_NAME=\"" $2 "\"";
    if ($1 == "wandb_project") print "WANDB_PROJECT=\"" $2 "\"";
}' "$CONFIG")

# Load .env variables securely (for wandb API key, etc.)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found."
fi

# Create run name with job info
NODE_ID=$(hostname | cut -d'.' -f1)
RUN_NAME="${BASE_RUN_NAME}_${SLURM_JOB_ID}"

# Use provided output directory or default
if [[ -n "$OUTPUT_DIR_ARG" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_ARG"
else
    OUTPUT_DIR="./output/${RUN_NAME}"
fi

# Set environment variables
export WANDB_PROJECT="${WANDB_PROJECT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# (6) Parse config into arguments
ARGS=$(awk -F': ' '
NF==2 && $1 !~ /^#/ && $1 != "" {
    gsub(/^[ \t]+|[ \t]+$/, "", $1);
    gsub(/^[ \t]+|[ \t]+$/, "", $2);
    gsub(/"/, "", $2);
    # Skip meta fields that we handle separately
    if ($1 != "base_run_name" && $1 != "wandb_project") {
        printf "--%s %s ", $1, $2
    }
}' "$CONFIG")

# Add additional arguments
FULL_ARGS="--deepspeed ${DEEPSPEED_CONFIG} --output_dir ${OUTPUT_DIR} --run_name ${RUN_NAME} ${ARGS}"

echo "Arguments: $FULL_ARGS"

# (7) Extract model name from config
MODEL_NAME=$(awk -F': ' '/^model_name_or_path:/ {gsub(/[ "\t]/,"",$2); print $2}' "$CONFIG")
echo "Model: $MODEL_NAME"

# (8) Clear corrupted cache and pre-download model on rank 0 only
# Use shared filesystem so all nodes can see the flag file
READY_FILE="$PWD/.model_ready_${SLURM_JOB_ID}.flag"

if [[ $SLURM_PROCID -eq 0 && -n "$MODEL_NAME" ]]; then
    echo "===================================="
    echo "Cleaning cache and pre-downloading model..."
    echo "===================================="
    
    # Clear potentially corrupted cache
    CACHE_DIR="$HOME/.cache/huggingface/hub/models--${MODEL_NAME//\/--}"
    if [[ -d "$CACHE_DIR" ]]; then
        echo "Removing cached model: $CACHE_DIR"
        rm -rf "$CACHE_DIR"
    fi
    
    # Pre-download model to ensure all files are present
    echo "Pre-downloading model: $MODEL_NAME"
    python -c "
import sys
try:
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    model_name = '$MODEL_NAME'
    print(f'Downloading model: {model_name}')
    AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print('✓ Model downloaded')
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print('✓ Tokenizer downloaded')
    AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print('✓ Processor downloaded')
    print('\\n✅ All artifacts ready')
except Exception as e:
    print(f'❌ Error downloading model: {e}')
    sys.exit(1)
" || { echo "Failed to download model"; exit 1; }
    
    echo "===================================="
    echo "Model ready, proceeding to training"
    echo "===================================="
    
    # Signal other ranks that download is complete
    touch "$READY_FILE"
else
    # Other ranks wait for rank 0 to finish downloading
    echo "[Rank $SLURM_PROCID] Waiting for model download to complete..."
    while [[ ! -f "$READY_FILE" ]]; do
        sleep 2
    done
    echo "[Rank $SLURM_PROCID] Model download complete, proceeding to training"
fi

echo "Starting training..."

# Launch distributed training
srun python -m torch.distributed.run \
    --nproc-per-node=$GPUSPERNODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $EXP_SCRIPT $FULL_ARGS

# Cleanup ready flag file
if [[ $SLURM_PROCID -eq 0 ]]; then
    rm -f "$READY_FILE"
fi

echo "=========================================="
echo "Finished configuration: $CONFIG"
echo "=========================================="
echo ""
