# SLURM Training Scripts for Qwen-VL

This directory contains SLURM job scripts and configuration files for training Qwen-VL models.

## Directory Structure

```
scripts/slurm/
├── slurm_sft.sh              # Main SLURM job script
├── configs/                  # Training configuration files
│   ├── pointing_qwen25.yaml  # Qwen2.5-VL pointing config (absolute coords)
│   └── pointing_qwen3.yaml   # Qwen3-VL pointing config (relative coords)
└── README.md                 # This file
```

## Quick Start

### 1. Configure Your Dataset

First, ensure your dataset is registered in `qwenvl/data/__init__.py`. Datasets should have:
- `annotation_path`: Path to the JSON/JSONL annotation file
- `data_path`: Path to the image folder (will be prepended to relative image paths in annotations)

Example:
```python
data_dict = {
    "pixmo_pointing_absolute": {
        "annotation_path": "/path/to/annotations.json",
        "data_path": "/path/to/images/"
    },
}
```

### 2. Choose or Create a Config File

We provide configs for:
- **Qwen2.5-VL** (`qwen25_affordance.yaml`): For training with absolute coordinates
- **Qwen3-VL** (`qwen3_affordance.yaml`): For training with relative coordinates

You can create your own config by copying one of these templates.

### 3. Launch Training

```bash
sbatch scripts/slurm/slurm_sft.sh configs/pointing_qwen25.yaml
```

Or with a custom output directory:

```bash
sbatch scripts/slurm/slurm_sft.sh configs/pointing_qwen25.yaml ./output/my_experiment
```

## Configuration File Format

Configuration files use YAML format. Key parameters:

### Model Configuration
```yaml
model_name_or_path: "Qwen/Qwen2.5-VL-7B-Instruct"  # Model to fine-tune
model_type: "qwen2.5vl"                             # qwen2vl, qwen2.5vl, or qwen3vl
```

### Dataset Configuration
```yaml
dataset_use: "handal_relative%100"  # Dataset name with optional sampling
```

**Dataset Sampling**: Use `%N` to sample N examples:
- `"dataset_name%100"` - Use only 100 samples
- `"dataset_name"` - Use entire dataset

**Multiple Datasets**: Separate with `+`:
- `"dataset1%100+dataset2%50"` - Mix 100 samples from dataset1 with 50 from dataset2

### Training Hyperparameters
```yaml
learning_rate: 2e-5                     # Learning rate
per_device_train_batch_size: 4          # Batch size per GPU
gradient_accumulation_steps: 4          # Gradient accumulation
num_train_epochs: 2.0                   # Number of epochs
```

### Image Preprocessing
```yaml
max_pixels: 50176  # Max pixels for image resize
min_pixels: 784    # Min pixels for image resize
```

### Model Tuning Options
```yaml
use_lora: False           # Use LoRA (parameter-efficient fine-tuning)
tune_mm_vision: False     # Fine-tune vision encoder
tune_mm_mlp: True         # Fine-tune vision-language projector
tune_mm_llm: True         # Fine-tune language model
```

### Weights & Biases Logging
```yaml
report_to: "wandb"
base_run_name: "Qwen25-Pointing-7B"
wandb_project: "qwen-pointing-training"
```

## SLURM Script Configuration

The main SLURM script (`slurm_sft.sh`) includes:

### Resource Allocation
```bash
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks-per-node=1    # Tasks per node
#SBATCH --gres=gpu:8           # Number of GPUs
#SBATCH --cpus-per-task=64     # CPUs per task
#SBATCH --mem=480G             # Memory per node
#SBATCH --time=48:00:00        # Maximum runtime
```

### Environment Setup
The script automatically:
1. Activates the `qwen-vl` conda environment
2. Detects available GPUs
3. Sets up distributed training variables
4. Parses the YAML config file
5. Launches training with `torchrun`

## Multi-Dataset Training

To train on multiple datasets:

1. Add all datasets to `qwenvl/data/__init__.py`:
```python
data_dict = {
    "dataset1": {
        "annotation_path": "/path/to/dataset1.json",
        "data_path": "/path/to/dataset1/images/"
    },
    "dataset2": {
        "annotation_path": "/path/to/dataset2.json",
        "data_path": "/path/to/dataset2/images/"
    },
}
```

2. In your config file:
```yaml
dataset_use: "dataset1%500+dataset2%300"
```

This will mix 500 samples from dataset1 with 300 samples from dataset2.

## Monitoring Training

### Check Job Status
```bash
squeue -u $USER
```

### View Logs
Logs are saved to:
- `slurm_logs/slurm-%j.out` (stdout)
- `slurm_logs/slurm-%j.err` (stderr)

Where `%j` is the job ID.

### Weights & Biases
Training metrics are logged to W&B if configured. Access your dashboard at:
```
https://wandb.ai/<your-entity>/<project-name>
```

## Output Structure

Training outputs are saved to the directory specified in the config or command line:

```
output_dir/
├── checkpoint-500/           # Saved checkpoints
│   ├── model.safetensors
│   ├── config.json
│   └── ...
├── checkpoint-1000/
└── ...
```

## Coordinate System Notes

- **Qwen2/3-VL**: Use relative coordinates (0-1000 scale)
- **Qwen2.5-VL**: Use absolute pixel coordinates

The data processor automatically handles coordinate adjustment based on model type and image resizing.


## Contact

For issues or questions, please refer to the main Qwen-VL repository documentation.
