import unittest

import numpy as np

from gpt2_streaming import GROUPS, measure_document, bootstrap_mean
from prepare_wikipedia_gpt2 import build_natural_input


class StreamingTests(unittest.TestCase):
  def test_query_distance_matched_uniform(self):
    n = 64
    a = np.tril(np.ones((n, n), dtype=np.float32))
    a /= np.arange(1, n + 1)[:, None]
    ids = [50256] + [5] * (n - 2) + [50256]
    ids[5], ids[30], ids[31] = 13, 198, 198
    tokens = [str(i) for i in ids]
    tokens[5], tokens[30], tokens[31] = ".", "Ċ", "Ċ"
    r = measure_document(dict(input_ids=ids, tokens=tokens,
                              boundary_positions=[30, 31]), a[None, None])
    np.testing.assert_allclose(r["past_mass"][0, 0], r["uniform_expected"], atol=1e-7)
    np.testing.assert_allclose(r["past_mass"], r["distance_expected"], atol=1e-7)
    self.assertAlmostEqual(r["normalized_entropy"][0, 0], 1, places=6)
    self.assertEqual(r["query_count"], 62)
    self.assertAlmostEqual(r["initial_removed_entropy"][0, 0], 1, places=6)

  def test_self_is_not_sink_and_future_does_not_affect_entropy(self):
    n = 64
    ids = [50256] + [13] * (n - 2) + [50256]
    tokens = ["<|endoftext|>"] + ["."] * (n - 2) + ["<|endoftext|>"]
    meta = dict(input_ids=ids, tokens=tokens, boundary_positions=[30, 31])
    a = np.eye(n, dtype=np.float32)
    a[0, 1:] = 100  # deliberately irrelevant future entries
    r = measure_document(meta, a[None, None])
    self.assertEqual(float(r["raw_entropy"][0, 0]), 0)
    self.assertEqual(float(r["self"][0, 0]), 1)
    np.testing.assert_array_equal(r["past_mass"], 0)

  def test_joined_tokenization_roundtrip(self):
    from transformers import GPT2TokenizerFast
    try:
      tokenizer = GPT2TokenizerFast.from_pretrained(
          "openai-community/gpt2", local_files_only=True)
    except OSError:
      self.skipTest("GPT-2 tokenizer is not cached")
    for maximum in [64, 1024]:
      r = build_natural_input(tokenizer, "Hello world. " * 100,
                              "Second paragraph. " * 100, maximum)
      self.assertLessEqual(len(r["input_ids"]), maximum)
      self.assertEqual(tokenizer.encode(r["retained_text"], add_special_tokens=False),
                       r["input_ids"][1:-1])
      self.assertEqual(r["separator_token_ids"], [198, 198])
      self.assertEqual(r["input_ids"].count(50256), 2)

  def test_bootstrap_constant(self):
    self.assertEqual(bootstrap_mean(np.ones(12), draws=100), (1, 1, 1))

  def test_previous_token_head_is_not_self_attention(self):
    n = 64
    ids = [50256] + [13] * (n - 2) + [50256]
    meta = dict(input_ids=ids, tokens=["."] * n, boundary_positions=[30, 31])
    a = np.eye(n, k=-1, dtype=np.float32)
    a[0, 0] = 1
    r = measure_document(meta, a[None, None])
    self.assertEqual(float(r["self"][0, 0]), 0)
    self.assertEqual(float(r["previous"][0, 0]), 1)
    self.assertEqual(float(r["raw_entropy"][0, 0]), 0)


if __name__ == "__main__":
  unittest.main()
