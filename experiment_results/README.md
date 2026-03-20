# Experiment Results

Track only small final summaries here.

Recommended:
- copy `results.md`
- copy `results.json`
- copy TTFT `summary.md`
- copy TTFT `summary.json`

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

TTFT example:

```bash
mkdir -p experiment_results/synthetic_ttft_table3_sweep_2026-03-20
cp outputs/synthetic_ttft_table3_sweep/summary.md \
  experiment_results/synthetic_ttft_table3_sweep_2026-03-20/
cp outputs/synthetic_ttft_table3_sweep/summary.json \
  experiment_results/synthetic_ttft_table3_sweep_2026-03-20/
```

If you want to keep one representative 32K point alongside the sweep summary, copy just the final markdown/json summary from that passage subdirectory instead of the full run directory:

```bash
mkdir -p experiment_results/synthetic_ttft_32k_2026-03-20
cp outputs/synthetic_ttft_table3_sweep/passage_32000/summary.md \
  experiment_results/synthetic_ttft_32k_2026-03-20/
cp outputs/synthetic_ttft_table3_sweep/passage_32000/summary.json \
  experiment_results/synthetic_ttft_32k_2026-03-20/
```
