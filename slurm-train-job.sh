#!/bin/bash
#SBATCH --job-name=gpt-l-l40s
#SBATCH --output=gpt-l-l40s%j.out
#SBATCH --error=gpt-l-l40s%j.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH -p viz
#SBATCH --gres=gpu:l40s:1

# Load the python module first
module load python3

# Change to the directory where this script was submitted
cd "$SLURM_SUBMIT_DIR"

# Create output directory with timestamp
OUTPUT_DIR="checkpoints/gpt-l-l40s"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Copy this script to output directory for reference
cp "$0" "$OUTPUT_DIR/training_job.sh"

# Run training and redirect all outputs to the directory
source /gpfs/automountdir/gpfs/homes/SEAS/home/g21775526/code/aladdin/.venv/bin/activate
module load python3

# FIX FOR LIBFFI ERROR: Inject host native paths
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# Create a private library folder for this job
LOCAL_LIB_DIR="$SLURM_SUBMIT_DIR/lib_compat"
mkdir -p "$LOCAL_LIB_DIR"

# Find whatever libffi version exists on this node and symlink it as libffi.so.6
if [ -f /usr/lib64/libffi.so.8 ]; then
    ln -sf /usr/lib64/libffi.so.8 "$LOCAL_LIB_DIR/libffi.so.6"
elif [ -f /usr/lib64/libffi.so.7 ]; then
    ln -sf /usr/lib64/libffi.so.7 "$LOCAL_LIB_DIR/libffi.so.6"
elif [ -f /usr/lib/x86_64-linux-gnu/libffi.so.8 ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.8 "$LOCAL_LIB_DIR/libffi.so.6"
elif [ -f /usr/lib/x86_64-linux-gnu/libffi.so.7 ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 "$LOCAL_LIB_DIR/libffi.so.6"
fi

# Force Python to read our spoofed directory first
export LD_LIBRARY_PATH="$LOCAL_LIB_DIR:$LD_LIBRARY_PATH"

# RUN TORCHRUN (Updated nproc_per_node=2 to utilize your requested gres layout)
python -m torch.distributed.run --standalone --nproc_per_node=1 models/salm.py \
    --output_dir "$OUTPUT_DIR" \
    --dataset "reddit-youtube" \
    > "$OUTPUT_DIR/py-output.txt" 2>&1

# Move SLURM output files to output directory
mv train.out "$OUTPUT_DIR/" 2>/dev/null || true
mv train.err "$OUTPUT_DIR/" 2>/dev/null || true

# Copy any generated logs or checkpoints
cp -r log*.txt "$OUTPUT_DIR/" 2>/dev/null || true
cp -r *.pt "$OUTPUT_DIR/" 2>/dev/null || true

echo "End time: $(date)"
echo "All outputs saved to: $OUTPUT_DIR"
