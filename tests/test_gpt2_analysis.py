import json
import math
import os
import tempfile
import types
import unittest

import numpy as np

from extract_attention_gpt2 import Example, extract_sharded, validate_attention
from gpt2_analysis import (
    compute_entropy_statistics,
    compute_token_sink_statistics,
    iter_attention_shards,
)
from prepare_wikipedia_gpt2 import build_input, truncate_around_boundary


EOS_ID = 50256


class CharacterTokenizer:
  """Offline offset-mapping fixture; real BPE has a separate regression test."""
  eos_token_id = EOS_ID
  name_or_path = "fake-tokenizer"
  init_kwargs = {"_commit_hash": "tokenizer-revision"}

  def encode(self, text, add_special_tokens=False):
    return [EOS_ID] if text == "bad" else [ord(c) for c in text]

  def __call__(self, text, **kwargs):
    ids = [ord(c) for c in text]
    if text.startswith("bad"):
      ids[0] = EOS_ID
    return dict(input_ids=ids, offset_mapping=[(i, i + 1) for i in range(len(text))])

  def convert_ids_to_tokens(self, ids):
    return [str(i) for i in ids]


def make_metadata(input_ids):
  return {
      "input_ids": input_ids,
      "tokens": ["<|endoftext|>" if token == EOS_ID else str(token)
                 for token in input_ids],
  }


def make_attention(rows):
  length = len(rows)
  attention = np.zeros((1, 1, length, length), dtype=np.float16)
  for query, row in enumerate(rows):
    attention[0, 0, query, :len(row)] = row
  return attention


class AnalysisTest(unittest.TestCase):

  def test_build_input_has_only_boundary_eos_and_natural_separator(self):
    tokenizer = CharacterTokenizer()
    result = build_input(tokenizer, "left", "right", max_length=9)
    self.assertEqual(
        result["input_ids"],
        [EOS_ID, ord('f'), ord('t'), 10, 10, ord('r'), ord('i'), EOS_ID])
    self.assertEqual(result["input_ids"].count(EOS_ID), 2)
    self.assertEqual(result["separator_token_ids"], [10, 10])
    self.assertEqual(result["paragraph_a_original_token_count"], 4)
    self.assertEqual(result["paragraph_b_original_token_count"], 5)
    with self.assertRaisesRegex(ValueError, "EOS"):
      build_input(tokenizer, "bad", "right", max_length=9)

  def test_truncate_preserves_boundary_sides(self):
    left, right = truncate_around_boundary(
        list(range(10)), list(range(100, 110)), 8)
    self.assertEqual(left, [6, 7, 8, 9])
    self.assertEqual(right, [100, 101, 102, 103])

    left, right = truncate_around_boundary([1, 2], list(range(10)), 8)
    self.assertEqual(left, [1, 2])
    self.assertEqual(right, list(range(6)))

  def test_sink_statistics_correct_frequency_and_exclude_eos(self):
    attention = make_attention([
        [1.0],
        [0.8, 0.2],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.3, 0.4],
    ])
    shards = [
        (make_metadata([EOS_ID, 5, 6, EOS_ID]), attention),
        (make_metadata([EOS_ID, 5, 7, EOS_ID]), attention),
    ]
    result = compute_token_sink_statistics(
        shards, EOS_ID, min_document_frequency=2,
        min_key_occurrences=2, min_average_attention_mass=0.01)
    token_five = [
        row for row in result["records"]
        if row["token_id"] == 5 and row["layer"] == 1
    ][0]
    self.assertAlmostEqual(token_five["attention_mass"], 1.4, places=3)
    self.assertAlmostEqual(token_five["uniform_expected_mass"], 2.0 / 3.0)
    self.assertAlmostEqual(token_five["uniform_enrichment"], 2.1, places=2)
    self.assertAlmostEqual(
        token_five["distance_enrichment"], 14.0 / 15.0, places=2)
    self.assertTrue(token_five["is_candidate"])
    self.assertTrue(all(not row["is_candidate"]
                        for row in result["eos_records"]))
    self.assertEqual(result["top_by_head"][(1, 1)][0]["token_id"], 5)

  def test_entropy_one_hot_and_causal_uniform(self):
    one_hot = make_attention([
        [1.0], [1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    ])
    uniform = make_attention([
        [1.0], [0.5, 0.5], [1 / 3, 1 / 3, 1 / 3],
        [0.25, 0.25, 0.25, 0.25]
    ])
    metadata = make_metadata([EOS_ID, 5, 6, EOS_ID])

    focused = compute_entropy_statistics([(metadata, one_hot)])
    broad = compute_entropy_statistics([(metadata, uniform)])
    self.assertAlmostEqual(float(focused["raw"][0, 0]), 0.0)
    self.assertAlmostEqual(float(focused["normalized"][0, 0]), 0.0)
    self.assertAlmostEqual(float(broad["normalized"][0, 0]), 1.0, places=3)
    expected_raw = (math.log(2) + math.log(3)) / 2.0
    self.assertAlmostEqual(float(broad["raw"][0, 0]), expected_raw, places=3)

    with_future_noise = uniform.copy()
    with_future_noise[0, 0, 1, 2:] = 9.0
    noisy = compute_entropy_statistics([(metadata, with_future_noise)])
    self.assertAlmostEqual(
        float(noisy["raw"][0, 0]), float(broad["raw"][0, 0]), places=6)

    changed_q_zero = uniform.copy()
    changed_q_zero[0, 0, 0, 0] = 0.5
    q_zero_excluded = compute_entropy_statistics([(metadata, changed_q_zero)])
    self.assertAlmostEqual(
        float(q_zero_excluded["raw"][0, 0]),
        float(broad["raw"][0, 0]), places=6)

  def test_validate_attention(self):
    valid = make_attention([
        [1.0], [0.5, 0.5], [0.2, 0.3, 0.5]
    ])
    row_error, future_max = validate_attention(valid)
    self.assertLessEqual(row_error, 1e-3)
    self.assertEqual(future_max, 0.0)

    invalid = valid.copy()
    invalid[0, 0, 0, 1] = 0.1
    invalid[0, 0, 0, 0] = 0.9
    with self.assertRaisesRegex(ValueError, "causal future"):
      validate_attention(invalid)

  def test_budgeted_shards_stop_before_limit(self):
    length = 64
    attention = np.zeros((1, 1, length, length), dtype=np.float16)
    for query in range(length):
      attention[0, 0, query, :query + 1] = 1.0 / (query + 1)

    class FakeExtractor:
      device = "cpu"
      model = types.SimpleNamespace(
          name_or_path="fake-gpt2",
          config=types.SimpleNamespace(_commit_hash="test-revision"))
      tokenizer = types.SimpleNamespace(name_or_path="fake-tokenizer")

      def get_attn_maps(self, examples):
        return [attention.copy() for _ in examples]

    examples = []
    for index in range(2):
      input_ids = [EOS_ID] + list(range(1, length - 1)) + [EOS_ID]
      examples.append(types.SimpleNamespace(
          input_ids=input_ids,
          tokens=[str(token) for token in input_ids],
          features={"sample_index": index}))

    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = os.path.join(temp_dir, "attention")
      budget = (1024 * 1024) + 12000
      manifest = extract_sharded(
          FakeExtractor(), examples, 2, output_dir,
          max_output_bytes=budget, stop_on_budget=True,
          max_sequence_length=64)
      self.assertEqual(manifest["processed_count"], 1)
      self.assertEqual(manifest["stop_reason"], "storage_budget")
      self.assertLessEqual(
          sum(os.path.getsize(os.path.join(root, filename))
              for root, _, files in os.walk(output_dir)
              for filename in files),
          budget)
      with open(os.path.join(output_dir, "manifest.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
      self.assertEqual(on_disk["processed_count"], 1)

  def test_two_record_sharded_analysis_smoke(self):
    tokenizer = CharacterTokenizer()
    records = []
    for sample_index in range(2):
      record = build_input(tokenizer, "a" * 30, "b" * 30, max_length=64)
      record["sample_index"] = sample_index
      records.append(record)
    examples = [Example(record, tokenizer) for record in records]

    class FakeExtractor:
      device = "cpu"
      model = types.SimpleNamespace(
          name_or_path="fake-gpt2",
          config=types.SimpleNamespace(_commit_hash="model-revision"))

      def __init__(self, tokenizer_instance):
        self.tokenizer = tokenizer_instance

      def get_attn_maps(self, batch):
        maps = []
        for example in batch:
          length = len(example.input_ids)
          attention = np.zeros(
              (1, 1, length, length), dtype=np.float16)
          for query in range(length):
            attention[0, 0, query, :query + 1] = 1.0 / (query + 1)
          maps.append(attention)
        return maps

    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = os.path.join(temp_dir, "attention")
      manifest = extract_sharded(
          FakeExtractor(tokenizer), examples, 2, output_dir,
          max_output_bytes=2 * 1024 * 1024,
          stop_on_budget=True, max_sequence_length=64)
      self.assertEqual(manifest["processed_count"], 2)
      self.assertEqual(manifest["tokenizer_revision"], "tokenizer-revision")

      entropy = compute_entropy_statistics(
          iter_attention_shards(output_dir))
      self.assertEqual(entropy["n_documents"], 2)
      self.assertAlmostEqual(float(entropy["normalized"][0, 0]), 1.0,
                             places=3)

      sinks = compute_token_sink_statistics(
          iter_attention_shards(output_dir), EOS_ID,
          min_document_frequency=2, min_key_occurrences=2,
          min_average_attention_mass=0.0)
      self.assertEqual(sinks["n_documents"], 2)
      self.assertTrue(sinks["top_global"])


if __name__ == "__main__":
  unittest.main()
