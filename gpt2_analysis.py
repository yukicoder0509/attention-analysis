"""Pure NumPy statistics for the GPT-2 general-attention notebook.

This module intentionally contains no plotting code.  The public notebook is
the analysis entry point; keeping the numerical routines here makes their
causal masking, denominators, and edge cases independently testable.
"""

import json
import math
import os

import numpy as np


DISTANCE_BUCKETS = (
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 16, "5-16"),
    (17, 64, "17-64"),
    (65, 256, "65-256"),
    (257, None, "257+"),
)

POSITION_BUCKETS = (
    (1, 16, "1-16"),
    (17, 32, "17-32"),
    (33, 64, "33-64"),
    (65, 128, "65-128"),
    (129, 256, "129-256"),
    (257, 512, "257-512"),
    (513, 1023, "513-1023"),
)


def _load_manifest(input_dir):
  manifest_path = os.path.join(input_dir, "manifest.json")
  with open(manifest_path, "r", encoding="utf-8") as manifest_file:
    return json.load(manifest_file)


def iter_attention_shards(input_dir, mmap_mode="r"):
  """Yields ``(metadata, attention)`` pairs in manifest order."""
  manifest = _load_manifest(input_dir)
  for shard in manifest.get("shards", []):
    metadata_path = os.path.join(input_dir, shard["metadata_file"])
    attention_path = os.path.join(input_dir, shard["attention_file"])
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
      metadata = json.load(metadata_file)
    yield metadata, np.load(attention_path, mmap_mode=mmap_mode)


def _distance_bucket_index(distance):
  for index, (lower, upper, _) in enumerate(DISTANCE_BUCKETS):
    if distance >= lower and (upper is None or distance <= upper):
      return index
  raise ValueError("distance must be a positive integer")


def _position_bucket_index(position):
  for index, (lower, upper, _) in enumerate(POSITION_BUCKETS):
    if lower <= position <= upper:
      return index
  return None


def _validate_shard(metadata, attention):
  input_ids = metadata["input_ids"]
  tokens = metadata["tokens"]
  if attention.ndim != 4:
    raise ValueError("attention must have shape [layers, heads, query, key]")
  if attention.shape[-1] != attention.shape[-2]:
    raise ValueError("attention query/key dimensions must be square")
  if len(input_ids) != attention.shape[-1] or len(tokens) != len(input_ids):
    raise ValueError("metadata length does not match attention map")
  return np.asarray(input_ids, dtype=np.int64), tokens


def compute_token_sink_statistics(shards, eos_token_id,
                                  min_document_frequency=2,
                                  min_key_occurrences=10,
                                  min_average_attention_mass=0.01):
  """Computes frequency- and distance-adjusted attention-to-token stats.

  Args:
    shards: iterable of ``(metadata, attention)`` pairs.
    eos_token_id: GPT-2's document-boundary token id.

  Returns:
    A dictionary containing flat CSV-friendly records, rankings, and distance
    diagnostics. Layer/head numbers in records are one-based.
  """
  token_stats = {}
  token_strings = {}
  distance_mass = None
  distance_link_count = np.zeros(len(DISTANCE_BUCKETS), dtype=np.int64)
  total_queries = 0
  n_documents = 0
  n_layers = n_heads = None

  for metadata, attention in shards:
    input_ids, tokens = _validate_shard(metadata, attention)
    layers, heads, sequence_length, _ = attention.shape
    if distance_mass is None:
      n_layers, n_heads = layers, heads
      distance_mass = np.zeros(
          (layers, heads, len(DISTANCE_BUCKETS)), dtype=np.float64)
    elif (layers, heads) != (n_layers, n_heads):
      raise ValueError("all shards must have the same layer/head dimensions")
    if sequence_length < 3:
      continue

    n_documents += 1
    query_start, query_stop = 1, sequence_length - 1
    total_queries += query_stop - query_start

    # Global attention-by-distance baseline. It is head-specific and excludes
    # self attention, just like the token statistics below.
    for query in range(query_start, query_stop):
      for bucket_index, (lower, upper, _) in enumerate(DISTANCE_BUCKETS):
        max_distance = query if upper is None else min(query, upper)
        if max_distance < lower:
          continue
        key_start = query - max_distance
        key_stop = query - lower + 1
        distance_mass[:, :, bucket_index] += np.asarray(
            attention[:, :, query, key_start:key_stop], dtype=np.float64
        ).sum(axis=-1)
        distance_link_count[bucket_index] += key_stop - key_start

    document_token_ids = set()
    for key in range(sequence_length - 1):
      token_id = int(input_ids[key])
      token_strings.setdefault(token_id, tokens[key])
      stat = token_stats.get(token_id)
      if stat is None:
        stat = {
            "mass": np.zeros((layers, heads), dtype=np.float64),
            "uniform": 0.0,
            "distance_links": np.zeros(
                len(DISTANCE_BUCKETS), dtype=np.int64),
            "key_occurrences": 0,
            "document_frequency": 0,
        }
        token_stats[token_id] = stat
      stat["key_occurrences"] += 1
      document_token_ids.add(token_id)

      first_query = max(query_start, key + 1)
      if first_query >= query_stop:
        continue
      stat["mass"] += np.asarray(
          attention[:, :, first_query:query_stop, key], dtype=np.float64
      ).sum(axis=-1)
      queries = np.arange(first_query, query_stop, dtype=np.int64)
      stat["uniform"] += np.sum(1.0 / (queries + 1.0))
      distances = queries - key
      for distance in distances:
        stat["distance_links"][_distance_bucket_index(int(distance))] += 1

    for token_id in document_token_ids:
      token_stats[token_id]["document_frequency"] += 1

  if n_documents < 2:
    raise ValueError(
        "at least two attention shards are required for token-sink analysis")

  distance_mean = np.divide(
      distance_mass,
      distance_link_count[np.newaxis, np.newaxis, :],
      out=np.zeros_like(distance_mass),
      where=distance_link_count[np.newaxis, np.newaxis, :] > 0)

  records = []
  eligible_records = []
  for token_id, stat in token_stats.items():
    distance_expected = np.sum(
        distance_mean * stat["distance_links"][np.newaxis, np.newaxis, :],
        axis=-1)
    average_mass = stat["mass"] / total_queries
    uniform_enrichment = np.divide(
        stat["mass"], stat["uniform"],
        out=np.zeros_like(stat["mass"]), where=stat["uniform"] > 0)
    distance_enrichment = np.divide(
        stat["mass"], distance_expected,
        out=np.zeros_like(stat["mass"]), where=distance_expected > 0)
    supported = (
        stat["document_frequency"] >= min_document_frequency
        and stat["key_occurrences"] >= min_key_occurrences)

    for layer in range(n_layers):
      for head in range(n_heads):
        is_candidate = bool(
            token_id != eos_token_id and supported
            and average_mass[layer, head] >= min_average_attention_mass)
        record = {
            "layer": layer + 1,
            "head": head + 1,
            "token_id": token_id,
            "token": token_strings.get(token_id, str(token_id)),
            "document_frequency": stat["document_frequency"],
            "key_occurrences": stat["key_occurrences"],
            "query_count": total_queries,
            "attention_mass": float(stat["mass"][layer, head]),
            "average_attention_mass": float(average_mass[layer, head]),
            "uniform_expected_mass": float(stat["uniform"]),
            "uniform_enrichment": float(uniform_enrichment[layer, head]),
            "distance_expected_mass": float(
                distance_expected[layer, head]),
            "distance_enrichment": float(
                distance_enrichment[layer, head]),
            "is_eos": token_id == eos_token_id,
            "is_candidate": is_candidate,
            "head_rank": None,
            "global_rank": None,
        }
        records.append(record)
        if is_candidate:
          eligible_records.append(record)

  ranking_key = lambda row: (
      row["distance_enrichment"], row["average_attention_mass"])
  top_by_head = {}
  for layer in range(1, n_layers + 1):
    for head in range(1, n_heads + 1):
      candidates = [r for r in eligible_records
                    if r["layer"] == layer and r["head"] == head]
      top_rows = sorted(
          candidates, key=ranking_key, reverse=True)[:3]
      for rank, row in enumerate(top_rows, start=1):
        row["head_rank"] = rank
      top_by_head[(layer, head)] = top_rows

  top_global = sorted(
      eligible_records, key=ranking_key, reverse=True)[:20]
  for rank, row in enumerate(top_global, start=1):
    row["global_rank"] = rank
  eos_records = [record for record in records if record["is_eos"]]
  return {
      "records": records,
      "top_by_head": top_by_head,
      "top_global": top_global,
      "eos_records": eos_records,
      "distance_mean": distance_mean,
      "distance_link_count": distance_link_count,
      "distance_labels": [bucket[2] for bucket in DISTANCE_BUCKETS],
      "n_documents": n_documents,
      "n_layers": n_layers,
      "n_heads": n_heads,
      "query_count": total_queries,
  }


def _entropy(probabilities):
  probabilities = np.asarray(probabilities, dtype=np.float64)
  log_probabilities = np.zeros_like(probabilities)
  np.log(probabilities, out=log_probabilities, where=probabilities > 0)
  return -(probabilities * log_probabilities).sum(axis=-1)


def compute_entropy_statistics(shards):
  """Computes document-balanced raw and causal-normalized head entropies."""
  document_raw = []
  document_normalized = []
  document_uniform = []
  trailing_raw = []
  trailing_normalized = []
  position_sum = None
  position_document_count = np.zeros(
      len(POSITION_BUCKETS), dtype=np.int64)
  position_query_count = np.zeros(
      len(POSITION_BUCKETS), dtype=np.int64)
  n_layers = n_heads = None
  total_queries = 0

  for metadata, attention in shards:
    _, _ = _validate_shard(metadata, attention)
    layers, heads, sequence_length, _ = attention.shape
    if position_sum is None:
      n_layers, n_heads = layers, heads
      position_sum = np.zeros(
          (layers, heads, len(POSITION_BUCKETS)), dtype=np.float64)
    elif (layers, heads) != (n_layers, n_heads):
      raise ValueError("all shards must have the same layer/head dimensions")
    if sequence_length < 3:
      continue

    raw_rows = []
    normalized_rows = []
    uniform_rows = []
    per_bucket = [[] for _ in POSITION_BUCKETS]
    for query in range(1, sequence_length - 1):
      visible = np.asarray(
          attention[:, :, query, :query + 1], dtype=np.float64)
      raw = _entropy(visible)
      uniform = math.log(query + 1)
      normalized = raw / uniform
      raw_rows.append(raw)
      normalized_rows.append(normalized)
      uniform_rows.append(uniform)
      total_queries += 1
      bucket_index = _position_bucket_index(query)
      if bucket_index is not None:
        per_bucket[bucket_index].append(normalized)
        position_query_count[bucket_index] += 1

    if not raw_rows:
      continue
    document_raw.append(np.mean(raw_rows, axis=0))
    document_normalized.append(np.mean(normalized_rows, axis=0))
    document_uniform.append(float(np.mean(uniform_rows)))

    for bucket_index, rows in enumerate(per_bucket):
      if rows:
        position_sum[:, :, bucket_index] += np.mean(rows, axis=0)
        position_document_count[bucket_index] += 1

    trailing_query = sequence_length - 1
    trailing_visible = np.asarray(
        attention[:, :, trailing_query, :trailing_query + 1],
        dtype=np.float64)
    trailing = _entropy(trailing_visible)
    trailing_raw.append(trailing)
    trailing_normalized.append(trailing / math.log(trailing_query + 1))

  if not document_raw:
    raise ValueError("no usable attention shards were provided")

  position_normalized = np.divide(
      position_sum,
      position_document_count[np.newaxis, np.newaxis, :],
      out=np.full_like(position_sum, np.nan),
      where=position_document_count[np.newaxis, np.newaxis, :] > 0)
  return {
      "raw": np.mean(document_raw, axis=0),
      "normalized": np.mean(document_normalized, axis=0),
      "uniform_raw": float(np.mean(document_uniform)),
      "trailing_eos_raw": np.mean(trailing_raw, axis=0),
      "trailing_eos_normalized": np.mean(
          trailing_normalized, axis=0),
      "position_normalized": position_normalized,
      "position_document_count": position_document_count,
      "position_query_count": position_query_count,
      "position_labels": [bucket[2] for bucket in POSITION_BUCKETS],
      "n_documents": len(document_raw),
      "n_layers": n_layers,
      "n_heads": n_heads,
      "query_count": total_queries,
  }
