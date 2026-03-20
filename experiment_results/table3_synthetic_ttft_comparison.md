# Table 3 Comparison: Synthetic 32K TTFT

This table compares the 32K setting from Table 3 in the original paper against the synthetic long-context benchmark in this repo.

| Setting | Total Length | Baseline TTFT (ms) | Block TTFT (ms) | Relative Reduction | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paper Table 3 (`TTFT-vanilla` vs `TTFT-block`) | 32K | 3638 | 45 | 98.7% | 80.84x |
| Reproduction (`Tulu3-RAG` vs `Tulu3-Block-FT`, precached) | ~32K | 1774.16 | 139.15 | 92.2% | 12.75x |
| Reproduction (`Tulu3-RAG` vs `Tulu3-Block-FT`, cache-ready) | ~32K | 1774.16 | 94.03 | 94.7% | 18.87x |

## Notes

- Paper numbers are from Table 3, 32K column: `TTFT-vanilla = 3638 ms`, `TTFT-block = 45 ms`.
- The paper states this setting assumes the KV states of retrieved passages have already been precomputed and cached in memory.
- `precached` in the reproduction excludes per-document KV construction but still includes online merged-cache preparation.
- `cache-ready` in the reproduction excludes both per-document KV construction and merged-cache preparation, so it is closer to a steady-state upper bound.
- The reproduction was run with shared attention backend `sdpa` on synthetic ~32K prompts, so it is not an exact replica of the paper's full inference stack.

## Sources

- Paper: [Block-Attention for Efficient Prefilling, Table 3](https://proceedings.iclr.cc/paper_files/paper/2025/file/a03037317560b8c5f2fb4b6466d4c439-Paper-Conference.pdf)
- Local benchmark summary: `outputs/synthetic_ttft_32k_gpu4/summary.md`
