# GPT-2 Experiment 2 and 3 results

> SUPERSEDED / DO NOT CITE THESE RESULTS. This legacy 158-document run
> independently encoded the paragraph separator, forcing ID 628 instead of
> the two ID-198 tokens produced by canonical tokenization of the joined text.
> Its claim that `ĊĊ` was a naturally tokenized boundary is withdrawn.
> See `../gpt2_streaming/experiment_summary.md` for the corrected 1000-document
> study, held-out validation, and the no-leading-EOT position diagnostic.
> Old dense `.npy` maps were deleted with user authorization; this report and
> the metadata remain only as an audit trail.

## Run manifest

- Wikipedia candidates: 1,000 (`wikimedia/wikipedia`, `20231101.en`, revision `ad5752b`, seed 42)
- Candidate JSONL SHA-256: `28d0cb0ef9ad1c68dac83a2304c15d24335c9991a19ca9b91b02b594c2447b54`
- GPT-2 snapshot: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Attention shards analyzed: 158; unprocessed: 842
- Storage: 2,143,709,412 bytes of the 2 GiB limit
- Stop reason: `storage_budget`
- Analyzed sequence lengths: 73–674 tokens
- Attention validation: maximum row-sum error `3.9864e-4`; maximum future mass `0`
- Ordinary query rows: 30,407

## Experiment 2: attention-sink candidates

EOT is excluded from candidate ranking. The top distance-adjusted head-token pairs are:

| Rank | Head | Token | Token ID | Documents | Occurrences | Mean mass | Uniform enrichment | Distance enrichment |
|---:|:---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | L03H07 | `.` | 13 | 148 | 960 | 0.13471 | 5.510 | 5.201 |
| 2 | L11H04 | `ĊĊ` | 628 | 158 | 158 | 0.02030 | 5.167 | 4.571 |
| 3 | L03H07 | `Ċ` | 198 | 104 | 549 | 0.07099 | 3.861 | 4.271 |
| 4 | L03H07 | `ĊĊ` | 628 | 158 | 158 | 0.01906 | 4.852 | 3.964 |
| 5 | L11H05 | `ĊĊ` | 628 | 158 | 158 | 0.01599 | 4.070 | 3.556 |
| 6 | L12H01 | `ĊĊ` | 628 | 158 | 158 | 0.01563 | 3.979 | 3.417 |
| 7 | L05H01 | `Ġby` | 416 | 86 | 151 | 0.01354 | 3.017 | 3.056 |
| 8 | L02H02 | `.` | 13 | 148 | 960 | 0.07459 | 3.051 | 2.947 |
| 9 | L02H12 | `.` | 13 | 148 | 960 | 0.02938 | 1.202 | 2.935 |
| 10 | L02H12 | `Ċ` | 198 | 104 | 549 | 0.01812 | 0.985 | 2.840 |

There are 667 eligible head-token pairs spanning 22 token IDs; 128 of 144 heads have at least one candidate under the fixed thresholds. `ĊĊ` is the naturally occurring two-newline paragraph boundary, not an inserted EOT.

The leading EOT baseline is much stronger than the ranked ordinary tokens. The largest value is L08H03, with mean attention mass 0.90818 and distance-adjusted enrichment 106.315. This baseline remains excluded from candidate ranking.

These are descriptive attention sinks / potential no-op candidates only. Attention alone does not establish functional no-op behavior.

## Experiment 3: causal entropy

| Layer | Raw entropy | Normalized entropy | Trailing-EOT normalized entropy |
|---:|---:|---:|---:|
| 1 | 2.5845 | 0.6169 | 0.5287 |
| 2 | 3.1706 | 0.7519 | 0.7351 |
| 3 | 2.1967 | 0.5396 | 0.3976 |
| 4 | 1.7018 | 0.4157 | 0.1650 |
| 5 | 1.6567 | 0.3994 | 0.1699 |
| 6 | 1.4009 | 0.3334 | 0.1441 |
| 7 | 1.6630 | 0.3957 | 0.1330 |
| 8 | 1.2331 | 0.2894 | 0.0700 |
| 9 | 1.6611 | 0.3928 | 0.1723 |
| 10 | 1.4207 | 0.3302 | 0.0984 |
| 11 | 1.5901 | 0.3679 | 0.1089 |
| 12 | 2.1456 | 0.5008 | 0.2827 |

The overall mean normalized entropy is 0.4445. Layer 2 is broadest on average (0.7519), while Layer 8 is most focused (0.2894). The broadest individual head is L01H12 (0.9500); the most focused is L05H12 (0.0020). The mean trailing-EOT normalized entropy is 0.2505.

Position-bucket normalized means are 0.4381, 0.4651, 0.4544, 0.4445, 0.4396, 0.4435, and 0.4216. The final `513–1023` bucket contains only 2 documents and 275 queries, so it should not be compared to earlier buckets without noting its much smaller support.

## Output files

- `token_sink_stats.csv`
- `entropy_by_head.csv`
- `entropy_by_position.npz`
- `General_Analysis_GPT2_executed.ipynb`
- PNG and PDF figures under `figures/`
