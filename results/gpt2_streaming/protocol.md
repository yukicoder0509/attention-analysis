# Frozen analysis protocol (before new model runs)

Primary reference: Clark et al. (2019), *What Does BERT Look At?*, Sections 3.2–3.3,
https://arxiv.org/abs/1906.04341, and the repository's General_Analysis.ipynb.

## Input audit and correction

The earlier pipeline encoded the two-newline separator independently. All 1000
stored sequences differ from canonical tokenization of the same decoded text.
`encode("\n\n")` gives ID 628; `encode("A\n\nB")` gives two ID-198 newline
tokens at the boundary. The earlier ID-628 finding is therefore not accepted as
evidence about naturally tokenized Wikipedia. Keep the same articles, paragraph
choices, seed and maximum length, but encode the joined text before truncation.
For a truncated sequence, retain paragraph-1's tail and paragraph-2's beginning.

## Runs and storage

1. Main run: all 1000 corrected samples, leading and trailing EOT, maximum 1024.
2. Position diagnostic: same 1000 corrected samples without the leading EOT;
   retain trailing EOT. Compare the attention to position zero and content.
   This changes both prefix identity and absolute positions and is not a clean
   causal test of token identity or of no-op functionality.
3. Retain only per-document sufficient statistics and a few illustration heads.
   Each dense attention tensor is consumed in RAM and released before the next
   sample. No accumulation of dense dumps. The old 158 shards may be deleted
   after their compact audit and legacy outputs have been preserved.

## Fixed discovery and validation partition

Articles 0–157 were previously examined. Treat them as discovery only. Articles
158–999 (842 articles) form validation. Do not select heads on validation results.
Plot full-corpus descriptive statistics; use only validation articles for the
following fixed tests. All layer/head labels are one-based.

| Hypothesis | Fixed quantity and threshold |
|---|---|
| H1 | L03H07 attention to periods exceeds query- and distance-bucket-matched expected mass |
| H2 | L11H04 attention to the natural paragraph boundary exceeds that matched expectation |
| H3 | L03H07 attention to newline keys exceeds that matched expectation |
| H4 | L08H03 mean ordinary-query attention to leading EOT exceeds 0.50 |
| H5 | L05H12 mean ordinary-query self attention exceeds 0.90 |
| H6 | Layer-2 mean causal-normalized entropy exceeds Layer-8 mean |
| H7 | For L01H12, over half the query rows at q >= 32 have max attention <= 0.10 |
| H8 | Without leading EOT, L08H03 mean attention to the first content token exceeds 0.50 |

For each test report effect size, article bootstrap 95% CI (10,000 draws, seed
42), fraction of positive article differences and a one-sided exact sign test.
Holm-correct all eight p-values as one family. The sign test tests the median
direction, while the bootstrap interval describes the mean effect. Report all
eight tests, including null/negative outcomes; no optional stopping or threshold
relaxation. Additional findings are explicitly descriptive/exploratory.

## Statistics and figures

- Figure 2 style: per-head points and per-layer means for initial EOT, natural
  paragraph boundary, and periods/commas; a second panel contrasts boundary-query
  and non-boundary-query attention. Report self-included mass for the original
  figure analogue and exclude self for candidate enrichment. Also compare only
  queries after the boundary to remove impossible future links.
- Frequency baseline: count of eligible earlier keys / (q+1).
- Distance baseline: for each query and head, distribute its observed mass in
  each relative-distance bucket uniformly over the keys in that bucket. Sum
  expected target mass over queries. Buckets: 1, 2–4, 5–16, 17–64, 65–256, 257+.
- Figure 4 style: raw entropy from ordinary queries and trailing EOT in separate
  panels, each with its own causal-uniform reference. Leading EOT has zero causal
  entropy by construction, so it cannot substitute for BERT's CLS summary query.
- Supplement normalized entropy, conditional entropy excluding the initial key,
  maximum attention, effective support and query-position-stratified statistics.
  Compute document means first, then average articles equally.
- Figure 1 style: real attention links for the fixed candidate heads using a
  disclosed sample/window; keep original weights and report omitted mass. No
  renormalization of a cropped attention submatrix.

The study remains attention-only. A statistically reliable attention pattern
does not establish a functional no-op, and BERT's reported gradient results are
not reproduced here. No BERT linguistic-head labels are assigned to GPT-2 heads.
