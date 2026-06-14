
OUTPUT_DIR="outputs/reddit-v1"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
cp "$0" "$OUTPUT_DIR/training_job.sh"
module load python3

# Run training with torchrun using detected GPU count
torchrun --standalone --nproc_per_node=4 model.py \
    --output_dir "$OUTPUT_DIR" \
    --dataset "reddit" \
    > "$OUTPUT_DIR/py-output.txt" 2>&1


mv small-gpt_${SLURM_JOB_ID}.out "$OUTPUT_DIR/" 2>/dev/null || true
mv small-gpt_${SLURM_JOB_ID}.err "$OUTPUT_DIR/" 2>/dev/null || true
cp -r log*.txt "$OUTPUT_DIR/" 2>/dev/null || true
cp -r *.pt "$OUTPUT_DIR/" 2>/dev/null || true

echo "End time: $(date)"
echo "All outputs saved to: $OUTPUT_DIR"
