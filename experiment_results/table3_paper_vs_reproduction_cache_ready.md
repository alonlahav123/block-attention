# Table 3 Reproduction Comparison

| Metric | 512 | 1K | 2K | 4K | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper TTFT-vanilla (ms) | 50 | 87 | 167 | 330 | 691 | 1515 | 3638 |
| Paper TTFT-block (ms) | 26 | 26 | 26 | 27 | 29 | 34 | 45 |
| Paper speedup (vanilla / block) | 1.92x | 3.35x | 6.42x | 12.22x | 23.83x | 44.56x | 80.84x |
| Reproduction Tulu3-RAG (ms) | 29.86 | 44.75 | 78.87 | 151.44 | 314.45 | 711.83 | 1794.22 |
| Reproduction Block-FT cache-ready (ms) | 23.34 | 23.47 | 23.51 | 26.21 | 36.41 | 57.30 | 95.67 |
| Reproduction speedup (RAG / block) | 1.28x | 1.91x | 3.35x | 5.78x | 8.64x | 12.42x | 18.76x |
