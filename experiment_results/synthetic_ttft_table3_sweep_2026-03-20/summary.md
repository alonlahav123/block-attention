# Synthetic TTFT Sweep

- This sweep follows the Table 3 setup style: fixed user input length with increasing retrieved passage length.
- Shared attention: `sdpa`
- User input tokens: `50`
- Documents: `8`
- Warmups per setting: `2`
- Measured runs per setting: `5`
- Reported TTFT values below use the median across measured runs.
- The local `32K` column uses a target passage length of `32000` tokens so the full prompt stays near the paper's 32K regime after prompt wrappers.

## Table 3 Style Comparison
| Path | 50 | 512 | 1K | 2K | 4K | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper TTFT-vanilla (ms) | 26 | 50 | 87 | 167 | 330 | 691 | 1515 | 3638 |
| Paper TTFT-block (ms) | 26 | 26 | 26 | 26 | 27 | 29 | 34 | 45 |
| Reproduction Tulu3-RAG (ms) | 18.86 | 29.86 | 44.75 | 78.87 | 151.44 | 314.45 | 711.83 | 1794.22 |
| Reproduction Tulu3-Block-FT precached (ms) | 43.18 | 43.26 | 43.34 | 42.85 | 46.31 | 59.49 | 87.38 | 141.85 |
| Reproduction Tulu3-Block-FT cache-ready (ms) | 22.97 | 23.34 | 23.47 | 23.51 | 26.21 | 36.41 | 57.30 | 95.67 |
| Reproduction speedup RAG / precached | 0.44x | 0.69x | 1.03x | 1.84x | 3.27x | 5.29x | 8.15x | 12.65x |
| Reproduction speedup RAG / cache-ready | 0.82x | 1.28x | 1.91x | 3.35x | 5.78x | 8.64x | 12.42x | 18.76x |

## Sweep Details
| Label | Target Passage Tokens | Actual Passage Tokens | RAG Prompt Tokens | Block Prompt Tokens | RAG Median TTFT (ms) | Block Precached Median TTFT (ms) | Block Cache-Ready Median TTFT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 50 | 98 | 319 | 316 | 18.86 | 43.18 | 22.97 |
| 512 | 512 | 537 | 758 | 754 | 29.86 | 43.26 | 23.34 |
| 1K | 1024 | 1046 | 1267 | 1263 | 44.75 | 43.34 | 23.47 |
| 2K | 2048 | 2077 | 2298 | 2294 | 78.87 | 42.85 | 23.51 |
| 4K | 4096 | 4127 | 4348 | 4344 | 151.44 | 46.31 | 26.21 |
| 8K | 8192 | 8220 | 8441 | 8437 | 314.45 | 59.49 | 36.41 |
| 16K | 16384 | 16412 | 16633 | 16629 | 711.83 | 87.38 | 57.30 |
| 32K | 32000 | 32040 | 32261 | 32257 | 1794.22 | 141.85 | 95.67 |

## Notes
- Paper values come from Table 3 of the Block-Attention paper.
- `precached` excludes per-document KV construction but still includes online merge-and-rotate.
- `cache-ready` excludes both per-document KV construction and merged-cache preparation, so it is the tighter steady-state upper bound.
- Each sweep point also writes a full per-setting summary under its own subdirectory.
