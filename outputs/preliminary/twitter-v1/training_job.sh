#!/bin/bash
#SBATCH --job-name=twitter-v1
#SBATCH --output=twitter-v1%j.out
#SBATCH --error=twitter-v1%j.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:4


# Load the python module first
module load python3

# Change to the directory where this script was submitted
cd "$SLURM_SUBMIT_DIR"

# Create output directory with timestamp
OUTPUT_DIR="outputs/twitter-v1"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Copy this script to output directory for reference
cp "$0" "$OUTPUT_DIR/training_job.sh"



# Run training and redirect all outputs to the directory
source /SEAS/home/g21775526/code/aladdin/.venv/bin/activate
module load python3

# torchrun --standalone  model.py \
#     --output_dir "$OUTPUT_DIR" \
#     --dataset "fineweb-twitter-reddit" \
#     > "$OUTPUT_DIR/py-output.txt" 2>&1


torchrun --standalone --nproc_per_node=4 model.py \
    --output_dir "$OUTPUT_DIR" \
    --dataset "twitter" \
    > "$OUTPUT_DIR/py-output.txt" 2>&1





# Move SLURM output files to output directory
mv train.out "$OUTPUT_DIR/" 2>/dev/null || true
mv train.err "$OUTPUT_DIR/" 2>/dev/null || true

# Copy any generated logs or checkpoints
cp -r log*.txt "$OUTPUT_DIR/" 2>/dev/null || true
cp -r *.pt "$OUTPUT_DIR/" 2>/dev/null || true

echo "End time: $(date)"
echo "All outputs saved to: $OUTPUT_DIR"
