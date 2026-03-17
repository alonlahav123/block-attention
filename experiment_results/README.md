# Experiment Results

Track only small final summaries here.

Recommended:
- copy `results.md`
- copy `results.json`

Do not track:
- `shard_runs/`
- `outputs/`
- per-chunk JSONL files
- logs
- downloaded models or datasets

Example:

```bash
mkdir -p experiment_results/tulu3_block_ft_eager_4gpu_2026-03-15
cp shard_runs/tulu3_block_ft_eager_4gpu_2026-03-15/results.md \
  experiment_results/tulu3_block_ft_eager_4gpu_2026-03-15/
cp shard_runs/tulu3_block_ft_eager_4gpu_2026-03-15/results.json \
  experiment_results/tulu3_block_ft_eager_4gpu_2026-03-15/
```
