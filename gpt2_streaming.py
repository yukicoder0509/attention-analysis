"""Streaming measurements for General_Analysis_GPT2.ipynb (no analysis CLI).

Stores per-article sufficient statistics, not dense all-head attention maps.
All head arrays have axes [layer, head]; article archives prepend [document].
"""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from gpt2_analysis import DISTANCE_BUCKETS, POSITION_BUCKETS


GROUPS = ("initial", "boundary", "period", "comma", "newline", "punct", "the")
FIXED_HEADS = ((2, 6), (10, 3), (7, 2), (4, 11), (0, 11))
DISCOVERY_COUNT = 158


def _mean(x, mask, axis=-1):
  if np.any(mask):
    return x[..., mask].mean(axis=axis, dtype=np.float64)
  return np.full(x.shape[:-1], np.nan)


def measure_document(metadata, attention, leading_eot=True):
  """Paper analogues plus query/distance-matched target baselines.

  Self-included category mass reproduces the Figure-2 measurement convention.
  Candidate statistics exclude self and EOT queries. For each query/head,
  expected group mass in a distance bucket equals bucket mass times the
  fraction of bucket keys belonging to that group. This controls query position
  as well as relative distance, unlike the earlier corpus-pooled baseline.
  """
  ids = np.asarray(metadata["input_ids"])
  tokens = metadata["tokens"]
  length = len(ids)
  if attention.shape[-2:] != (length, length):
    raise ValueError("attention shape does not match input")
  a = np.asarray(attention, dtype=np.float32)
  indices = np.arange(length)
  causal = indices[:, None] >= indices[None, :]
  a = np.where(causal, a, 0)
  # Re-normalize only causal support to remove float16 storage rounding.
  a /= a.sum(axis=-1, keepdims=True)
  ordinary = ids != 50256
  ordinary[0] = False  # q=0 has degenerate support in both input conditions
  late = ordinary & (indices >= 32)
  boundary = np.zeros(length, dtype=bool)
  boundary[metadata["boundary_positions"]] = True
  period = np.asarray([t.lstrip("Ġ") == "." for t in tokens])
  comma = np.asarray([t.lstrip("Ġ") == "," for t in tokens])
  newline = np.asarray(["Ċ" in t for t in tokens])
  initial = indices == 0
  masks = np.stack((initial, boundary, period, comma, newline,
                    period | comma, ids == 262))
  # a[..., q, k] @ masks[k, group] -> [layer, head, q, group]
  mass = np.einsum("lhqk,ck->lhqc", a, masks.astype(np.float32), optimize=True)
  diagonal = np.diagonal(a, axis1=-2, axis2=-1)
  past_mass = mass - diagonal[..., None] * masks.T
  uniform_rows = np.cumsum(masks, axis=1).T / (indices[:, None] + 1)
  uniform_past = uniform_rows - masks.T / (indices[:, None] + 1)
  expected = np.zeros_like(mass)
  for lo, hi, _ in DISTANCE_BUCKETS:
    distances = indices[:, None] - indices[None, :]
    bucket = (distances >= lo) & (distances <= (length if hi is None else hi))
    count = bucket.sum(-1)
    target_count = bucket.astype(np.float32) @ masks.T.astype(np.float32)
    fractions = np.divide(target_count, count[:, None],
                          out=np.zeros_like(target_count), where=count[:, None] > 0)
    bucket_mass = np.einsum("lhqk,qk->lhq", a, bucket, optimize=True)
    expected += bucket_mass[..., None] * fractions
  log_a = np.zeros_like(a)
  np.log(a, out=log_a, where=a > 0)
  entropy_rows = -(a * log_a).sum(-1, dtype=np.float64)
  uniform_entropy = np.log(indices + 1)
  norm_rows = np.divide(entropy_rows, uniform_entropy,
                        out=np.zeros_like(entropy_rows), where=uniform_entropy > 0)
  first = a[..., 0]
  rest = 1 - first
  first_term = np.zeros_like(first)
  np.log(first, out=first_term, where=first > 0)
  first_term *= first
  conditional = np.divide(entropy_rows + first_term, rest,
                          out=np.zeros_like(entropy_rows), where=rest > 1e-6)
  conditional += np.log(np.maximum(rest, 1e-30))
  conditional_norm = np.divide(conditional, np.log(np.maximum(indices, 1)),
                               out=np.zeros_like(conditional), where=indices > 1)
  conditional_norm = np.where((rest > 1e-6) & (indices > 1), conditional_norm, np.nan)
  maximum = a.max(-1)
  boundary_queries = boundary
  other_queries = ordinary & ~boundary
  exposed = ordinary & (indices > np.max(np.flatnonzero(boundary)))
  after32 = exposed & (indices >= np.max(np.flatnonzero(boundary)) + 32)
  result = {
      "category_all": mass.mean(axis=2, dtype=np.float64),
      "category_ordinary": mass[:, :, ordinary].mean(axis=2, dtype=np.float64),
      "past_mass": past_mass[:, :, ordinary].mean(axis=2, dtype=np.float64),
      "distance_expected": expected[:, :, ordinary].mean(axis=2, dtype=np.float64),
      "uniform_expected": uniform_past[ordinary].mean(axis=0),
      "boundary_from_boundary": _mean(mass[..., 1], boundary_queries),
      "boundary_from_other": _mean(mass[..., 1], other_queries),
      "boundary_from_exposed": _mean(past_mass[..., 1], exposed),
      "boundary_from_far": _mean(past_mass[..., 1], after32),
      "boundary_to_initial": _mean(first, boundary_queries),
      "boundary_self": _mean(diagonal, boundary_queries),
      "initial": _mean(first, ordinary),
      "self": _mean(diagonal, ordinary),
      "previous": _mean(np.concatenate((np.zeros((*a.shape[:2], 1)),
                  a[:, :, indices[1:], indices[:-1]]), axis=-1), ordinary),
      "raw_entropy": _mean(entropy_rows, ordinary),
      "normalized_entropy": _mean(norm_rows, ordinary),
      "all_raw_entropy": entropy_rows.mean(-1),
      "trailing_raw_entropy": entropy_rows[..., -1],
      "trailing_normalized_entropy": norm_rows[..., -1],
      "late_normalized_entropy": _mean(norm_rows, late),
      "late_max_attention": _mean(maximum, late),
      "late_broad_fraction": _mean((maximum <= 0.10).astype(float), late),
      "late_effective_support": _mean(np.exp(entropy_rows), late),
      "initial_removed_entropy": np.nanmean(conditional_norm[..., late], axis=-1),
      "uniform_raw": uniform_entropy[ordinary].mean(),
      "all_uniform_raw": uniform_entropy.mean(),
      "trailing_uniform_raw": uniform_entropy[-1],
      "query_count": ordinary.sum(),
      "length": length,
  }
  result["position_entropy"] = np.stack([
      _mean(norm_rows, ordinary & (indices >= lo) & (indices <= hi))
      for lo, hi, _ in POSITION_BUCKETS], axis=-1)
  result["position_counts"] = np.asarray([
      np.sum(ordinary & (indices >= lo) & (indices <= hi))
      for lo, hi, _ in POSITION_BUCKETS])
  # Fixed period-head probe: recent versus older period preference.
  last_period = np.maximum.accumulate(np.where(period, indices, -1))
  last_period = np.concatenate(([-1], last_period[:-1]))
  has_period = ordinary & (last_period >= 0)
  chosen = a[2, 6] if a.shape[:2] == (12, 12) else a[0, 0]
  argmax = chosen.argmax(-1)
  result["period_head_latest_mass"] = (
      chosen[indices[has_period], last_period[has_period]].mean()
      if has_period.any() else np.nan)
  result["period_head_argmax_period"] = (
      period[argmax[has_period]].mean() if has_period.any() else np.nan)
  result["period_head_argmax_latest"] = (
      (argmax[has_period] == last_period[has_period]).mean()
      if has_period.any() else np.nan)
  return result


def stream_measurements(input_file, output_dir, model_name="openai-community/gpt2",
                        variants=("canonical", "no_leading_eot"), limit=None):
  """Called from the notebook. Reuse completed archives; never retain dense maps."""
  from extract_attention_gpt2 import (AttnMapExtractor, Example, validate_attention,
                                     resolve_hf_revision)
  import torch
  import transformers
  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)
  raw = Path(input_file).read_bytes()
  rows = [json.loads(line) for line in raw.splitlines()]
  if limit is not None:
    rows = rows[:limit]
  digest = hashlib.sha256(raw).hexdigest()
  manifest_path = output / "stream_manifest.json"
  saved = None
  if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    if saved["input_sha256"] != digest:
      raise ValueError("output belongs to a different input dataset")
    if saved["model_name"] != model_name or saved["candidate_count"] != len(rows):
      raise ValueError("output belongs to a different model or document count")
  extractor = None
  manifest = {"input_file": str(input_file), "input_sha256": digest,
              "candidate_count": len(rows), "discovery_count": min(158, len(rows)),
              "validation_count": max(0, len(rows) - 158),
              "model_name": model_name, "variants": {},
              "storage_mode": "one dense map in RAM, compact per-document statistics on disk",
              "groups": list(GROUPS), "python": platform.python_version(),
              "numpy": np.__version__, "torch": str(torch.__version__),
              "transformers": transformers.__version__}
  if saved is not None:
    manifest = saved  # Preserve provenance of already completed variants.
  for variant in variants:
    archive_path = output / (variant + "_documents.npz")
    if archive_path.exists():
      if saved is None or variant not in manifest["variants"]:
        raise ValueError("archive is missing its completed-run manifest entry")
      with np.load(archive_path) as archive:
        if len(archive["length"]) != len(rows):
          raise ValueError("existing archive has a different document count")
        if list(archive["article_ids"]) != [row["article_id"] for row in rows]:
          raise ValueError("existing archive has different articles or order")
      print("Reusing", archive_path, flush=True)
      continue
    if extractor is None:
      extractor = AttnMapExtractor(model_name)
      manifest["model_revision"] = resolve_hf_revision(model_name, "config.json")
      manifest["tokenizer_revision"] = resolve_hf_revision(model_name, "tokenizer_config.json")
      manifest["device"] = extractor.device
      manifest["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    start = time.monotonic()
    documents = []
    max_bytes, max_error, max_future = 0, 0, 0
    for i, row in enumerate(rows):
      metadata = dict(row)
      if variant == "no_leading_eot":
        metadata["input_ids"] = row["input_ids"][1:]
        metadata["tokens"] = row["tokens"][1:]
        metadata["boundary_positions"] = [q - 1 for q in row["boundary_positions"]]
      elif variant != "canonical":
        raise ValueError("unknown variant " + variant)
      attention = extractor.get_attn_maps([Example(metadata, extractor.tokenizer)])[0]
      error, future = validate_attention(attention)
      max_bytes = max(max_bytes, attention.nbytes)
      max_error, max_future = max(error, max_error), max(future, max_future)
      documents.append(measure_document(metadata, attention, variant == "canonical"))
      if i == 0:
        illustration = attention[tuple(zip(*FIXED_HEADS))].copy()
        np.savez_compressed(output / (variant + "_example_heads.npz"),
                            attention=illustration, heads=FIXED_HEADS,
                            tokens=np.asarray(metadata["tokens"]),
                            input_ids=np.asarray(metadata["input_ids"]),
                            boundary_positions=metadata["boundary_positions"])
      del attention
      if (i + 1) % 50 == 0 or i + 1 == len(rows):
        print("{}: {}/{} documents; {:.1f}s; no dense dump retained".format(
            variant, i + 1, len(rows), time.monotonic() - start), flush=True)
    arrays = {key: np.asarray([d[key] for d in documents]) for key in documents[0]}
    arrays["article_ids"] = np.asarray([row["article_id"] for row in rows])
    np.savez_compressed(archive_path, **arrays)
    manifest["variants"][variant] = {
        "processed_count": len(rows), "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "max_dense_array_bytes": max_bytes, "row_sum_max_error": max_error,
        "future_attention_max": max_future,
        "elapsed_seconds": time.monotonic() - start}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    del documents, arrays
  runs = {}
  for variant in variants:
    with np.load(output / (variant + "_documents.npz")) as archive:
      runs[variant] = dict(archive)
  return runs


def bootstrap_mean(values, seed=42, draws=10000):
  values = np.asarray(values, dtype=np.float64)
  values = values[np.isfinite(values)]
  if len(values) == 0:
    return np.nan, np.nan, np.nan
  rng = np.random.default_rng(seed)
  means = np.concatenate([
      values[rng.integers(len(values), size=(min(250, draws - i), len(values)))].mean(1)
      for i in range(0, draws, 250)])
  low, high = np.quantile(means, [0.025, 0.975])
  return float(values.mean()), float(low), float(high)


def validation_tests(main, control, discovery_count=DISCOVERY_COUNT):
  """Eight pre-specified article sign tests with family-wise Holm correction."""
  from scipy.stats import binomtest
  q = slice(discovery_count, None)
  period, boundary, newline = (GROUPS.index(k) for k in ("period", "boundary", "newline"))
  tests = [
      ("H1", "L03H07 period excess over matched distance", main["past_mass"][q, 2, 6, period] - main["distance_expected"][q, 2, 6, period]),
      ("H2", "L11H04 natural boundary excess over matched distance", main["past_mass"][q, 10, 3, boundary] - main["distance_expected"][q, 10, 3, boundary]),
      ("H3", "L03H07 newline excess over matched distance", main["past_mass"][q, 2, 6, newline] - main["distance_expected"][q, 2, 6, newline]),
      ("H4", "L08H03 initial EOT mass minus 0.50", main["initial"][q, 7, 2] - 0.50),
      ("H5", "L05H12 self mass minus 0.90", main["self"][q, 4, 11] - 0.90),
      ("H6", "Layer 2 minus Layer 8 normalized entropy", main["normalized_entropy"][q, 1].mean(-1) - main["normalized_entropy"][q, 7].mean(-1)),
      ("H7", "L01H12 broad-query fraction minus 0.50 (q>=32)", main["late_broad_fraction"][q, 0, 11] - 0.50),
      ("H8", "L08H03 first-content mass minus 0.50 without EOT", control["initial"][q, 7, 2] - 0.50),
  ]
  records = []
  for name, description, values in tests:
    values = values[np.isfinite(values)]
    mean, lo, hi = bootstrap_mean(values)
    n_positive, n_nonzero = int(np.sum(values > 0)), int(np.sum(values != 0))
    p = binomtest(n_positive, n_nonzero, 0.5, alternative="greater").pvalue if n_nonzero else 1.0
    records.append(dict(hypothesis=name, description=description, n=len(values),
                        mean_effect=mean, ci_low=lo, ci_high=hi,
                        positive_documents=n_positive, nonzero_documents=n_nonzero,
                        positive_fraction=n_positive / max(n_nonzero, 1), p_value=p))
  order = np.argsort([r["p_value"] for r in records])
  running = 0.0
  for rank, index in enumerate(order):
    running = max(running, (len(records) - rank) * records[index]["p_value"])
    records[index]["holm_p"] = min(1.0, running)
    records[index]["supported"] = running < 0.05 and records[index]["ci_low"] > 0
  return records


def export_tables(main, control, output_dir):
  """Compact, auditable tables used by the notebook figures and report."""
  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)

  def write(name, records):
    with (output / name).open("w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=list(records[0]))
      writer.writeheader()
      writer.writerows(records)

  tests = validation_tests(main, control)
  write("validation_tests.csv", tests)
  records = []
  for l in range(12):
    for h in range(12):
      record = dict(layer=l + 1, head=h + 1, documents=len(main["length"]))
      for key in ("initial", "self", "previous", "raw_entropy", "normalized_entropy",
                  "trailing_raw_entropy", "trailing_normalized_entropy",
                  "late_max_attention", "late_broad_fraction", "late_effective_support",
                  "initial_removed_entropy", "boundary_from_boundary",
                  "boundary_from_other", "boundary_from_exposed", "boundary_from_far"):
        record[key] = float(np.nanmean(main[key][:, l, h]))
      record["first_content_without_eot"] = float(control["initial"][:, l, h].mean())
      records.append(record)
  write("head_statistics.csv", records)
  entropy_records = []
  for record in records:
    entropy_records.append({
        key: record[key] for key in ("layer", "head", "documents", "raw_entropy",
              "normalized_entropy", "trailing_raw_entropy", "trailing_normalized_entropy")})
    entropy_records[-1]["query_count"] = int(main["query_count"].sum())
  write("entropy_by_head.csv", entropy_records)
  np.savez_compressed(output / "entropy_by_position.npz",
      normalized=np.nanmean(main["position_entropy"], axis=0),
      document_count=(main["position_counts"] > 0).sum(0),
      query_count=main["position_counts"].sum(0),
      bin_labels=np.asarray([label for _, _, label in POSITION_BUCKETS]))
  records = []
  for split, selection in (("all", slice(None)), ("discovery", slice(0, 158)), ("validation", slice(158, None))):
    actual = main["past_mass"][selection].mean(0)
    expected = main["distance_expected"][selection].mean(0)
    uniform = main["uniform_expected"][selection].mean(0)
    for l in range(12):
      for h in range(12):
        for c, group in enumerate(GROUPS):
          records.append(dict(split=split, layer=l + 1, head=h + 1, group=group,
              attention_mass=actual[l, h, c], distance_expected=expected[l, h, c],
              uniform_expected=uniform[c],
              distance_enrichment=actual[l, h, c] / max(expected[l, h, c], 1e-30),
              uniform_enrichment=actual[l, h, c] / max(uniform[c], 1e-30)))
  write("category_enrichment.csv", records)
  return tests
