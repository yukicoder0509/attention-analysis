"""Runs GPT-2 over input data and writes out its attention maps to disk.

This is the GPT-2 counterpart of extract_attention.py (which runs BERT). Its
legacy mode produces pickle files in the same format expected by the original
analysis notebooks and by head_distances.py:

    {
        "tokens": list of strings,
        "attns":  [n_layers, n_heads, n_tokens, n_tokens] float16 tensor
    }

For the Wikipedia experiments, pass the exact ``input_ids`` JSONL produced by
prepare_wikipedia_gpt2.py and use ``--output-dir``. That mode writes validated,
float16 NumPy shards plus a manifest while enforcing an optional byte budget.

Legacy free-text inputs retain the older BERT-like boundary substitution:

  * Special tokens. GPT-2 has no [CLS]/[SEP]. Following the port design, every
    BERT special token is substituted with GPT-2's document-boundary token
    <|endoftext|>. A single text segment becomes

        <|endoftext|> A <|endoftext|>

    and a pair of segments becomes

        <|endoftext|> A <|endoftext|> B <|endoftext|>

    The prepared Wikipedia path does not use this inter-segment boundary. Its
    input is ``<eot> paragraph_1 \n\n paragraph_2 <eot>``.

  * Causal attention. GPT-2 is a decoder, so attns[i, j] == 0 for j > i (a
    token cannot attend to future tokens). This is an architectural fact, not a
    bug; analyses about "next token" / trailing separators are expected to be
    empty. See the README for which notebook sections remain meaningful.

Word-level attention (bpe_utils / --word_level in the BERT extractor) is NOT
supported here: it is hard-wired to WordPiece + [CLS]/[SEP] and does not
transfer to GPT-2's byte-level BPE.
"""

import argparse
import json
import os
import pickle
import platform

import numpy as np
import torch
import transformers
from transformers import GPT2Model, GPT2TokenizerFast
from transformers.utils.hub import cached_file, extract_commit_hash


BOUNDARY_TOKEN = "<|endoftext|>"


MANIFEST_RESERVE_BYTES = 1024 * 1024


def resolve_hf_revision(model_name, filename):
  """Returns the commit encoded in a locally resolved HF snapshot path."""
  try:
    resolved_file = cached_file(
        model_name, filename, local_files_only=True)
  except (OSError, ValueError):
    return None
  return extract_commit_hash(resolved_file, None)


def load_records(path):
  """Loads either a JSON list or newline-delimited JSON records."""
  with open(path, "r", encoding="utf-8") as f:
    if path.endswith(".jsonl"):
      return [json.loads(line) for line in f if line.strip()]
    return json.load(f)


def write_pickle(o, path):
  with open(path, "wb") as f:
    pickle.dump(o, f, -1)


def write_json(o, path):
  with open(path, "w", encoding="utf-8") as f:
    json.dump(o, f, ensure_ascii=False, indent=2, sort_keys=True)


class Example(object):
  """A single input sequence to be passed into GPT-2.

  Accepts the same feature dicts as extract_attention.py's Example, plus an
  optional two-segment form:
    * {"text": "..."}                 -> <eot> text <eot>
    * {"words": ["w1", "w2", ...]}    -> <eot> w1 w2 ... <eot>
    * {"text_a": "...", "text_b": ...}-> <eot> a <eot> b <eot>
    * {"tokens": ["<|endoftext|>", ...]} -> used as-is (already boundary-wrapped)
  """

  def __init__(self, features, tokenizer):
    self.features = features
    eot_id = tokenizer.eos_token_id

    if "input_ids" in features:
      self.input_ids = [int(token_id) for token_id in features["input_ids"]]
      expected_tokens = tokenizer.convert_ids_to_tokens(self.input_ids)
      if "tokens" in features and features["tokens"] != expected_tokens:
        raise ValueError("tokens do not match the supplied input_ids")
      self.tokens = expected_tokens
      return

    if "tokens" in features:
      # Pre-tokenized (GPT-2 token strings, boundaries already inserted).
      self.tokens = features["tokens"]
      self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
      return

    if "text_a" in features:
      segments = [features["text_a"], features.get("text_b")]
    elif "text" in features:
      segments = [features["text"]]
    else:
      segments = [" ".join(features["words"])]
    segments = [s for s in segments if s is not None and s != ""]

    # Build: <eot> seg0 <eot> seg1 <eot> ...  (substitutes [CLS]/[SEP]).
    input_ids = [eot_id]
    for segment in segments:
      input_ids.extend(tokenizer.encode(segment))
      input_ids.append(eot_id)

    self.input_ids = input_ids
    # Human-readable token strings; the boundary id decodes to BOUNDARY_TOKEN.
    self.tokens = tokenizer.convert_ids_to_tokens(input_ids)


def examples_in_batches(examples, batch_size):
  for i in range(1 + ((len(examples) - 1) // batch_size)):
    yield examples[i * batch_size:(i + 1) * batch_size]


class AttnMapExtractor(object):
  """Runs GPT-2 over examples to get its attention maps."""

  def __init__(self, model_name="gpt2", device=None):
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    self.model = GPT2Model.from_pretrained(
        model_name, output_attentions=True, attn_implementation="eager")
    self.model.eval()
    self.model.to(self.device)
    self.pad_id = self.tokenizer.eos_token_id  # masked out via attention_mask

  @torch.no_grad()
  def get_attn_maps(self, examples):
    """Returns a list of [n_layers, n_heads, seq_len, seq_len] float16 arrays,
    one per example, each already sliced to the example's true length."""
    seq_lens = [len(e.input_ids) for e in examples]
    batch_max = max(seq_lens)

    input_ids = torch.full((len(examples), batch_max), self.pad_id,
                           dtype=torch.long)
    attention_mask = torch.zeros((len(examples), batch_max), dtype=torch.long)
    for i, e in enumerate(examples):
      n = len(e.input_ids)
      input_ids[i, :n] = torch.tensor(e.input_ids, dtype=torch.long)
      attention_mask[i, :n] = 1

    outputs = self.model(
        input_ids=input_ids.to(self.device),
        attention_mask=attention_mask.to(self.device),
        use_cache=False)

    if outputs.attentions is None:
      raise RuntimeError(
          "model returned no attention weights; eager attention is required")

    results = []
    for i, seq_len in enumerate(seq_lens):
      # [n_layers, n_heads, seq_len, seq_len]; right-padding + causal mask means
      # the real tokens' rows/cols are unaffected by the padded positions.
      e_attn = np.empty(
          (len(outputs.attentions), outputs.attentions[0].shape[1],
           seq_len, seq_len), dtype=np.float16)
      for layer, layer_attention in enumerate(outputs.attentions):
        e_attn[layer] = layer_attention[
            i, :, :seq_len, :seq_len].detach().cpu().numpy()
      results.append(e_attn)
    return results


def validate_attention(attention):
  """Validates normalization and GPT-2's causal upper triangle."""
  if attention.ndim != 4 or attention.shape[-1] != attention.shape[-2]:
    raise ValueError(
        "attention must have shape [layers, heads, sequence, sequence]")
  row_sums = attention.sum(axis=-1, dtype=np.float32)
  row_sum_error = float(np.max(np.abs(row_sums - 1.0)))
  future_max = 0.0
  for query in range(attention.shape[-1] - 1):
    future_max = max(
        future_max,
        float(np.max(attention[:, :, query, query + 1:])))
  if row_sum_error > 1e-3:
    raise ValueError(
        "attention rows do not sum to one (max error {:.6g})".format(
            row_sum_error))
  if future_max > 1e-6:
    raise ValueError(
        "causal future attention exceeds tolerance ({:.6g})".format(
            future_max))
  return row_sum_error, future_max


def directory_size(path):
  total = 0
  if not os.path.isdir(path):
    return total
  for root, _, filenames in os.walk(path):
    for filename in filenames:
      total += os.path.getsize(os.path.join(root, filename))
  return total


def extract_sharded(extractor, examples, candidate_count, output_dir,
                    max_output_bytes=None, stop_on_budget=False,
                    max_sequence_length=1024):
  """Writes validated per-example attention shards and a run manifest."""
  os.makedirs(output_dir, exist_ok=True)
  if os.listdir(output_dir):
    raise ValueError("output directory must be empty: " + output_dir)
  if stop_on_budget and max_output_bytes is None:
    raise ValueError("--stop-on-budget requires --max-output-bytes")

  resolved_revision = (
      getattr(extractor.model.config, "_commit_hash", None)
      or resolve_hf_revision(extractor.model.name_or_path, "config.json"))
  tokenizer_revision = (
      getattr(extractor.tokenizer, "init_kwargs", {}).get("_commit_hash")
      or resolve_hf_revision(
          extractor.tokenizer.name_or_path, "tokenizer_config.json"))
  manifest = {
      "format_version": 1,
      "model_name": extractor.model.name_or_path,
      "model_revision": resolved_revision,
      "tokenizer_name": extractor.tokenizer.name_or_path,
      "tokenizer_revision": tokenizer_revision,
      "eos_token_id": getattr(extractor.tokenizer, "eos_token_id", None),
      "candidate_count": candidate_count,
      "eligible_count": len(examples),
      "processed_count": 0,
      "unprocessed_count": candidate_count,
      "skipped_over_length": candidate_count - len(examples),
      "max_sequence_length": max_sequence_length,
      "max_output_bytes": max_output_bytes,
      "stop_on_budget": stop_on_budget,
      "stop_reason": "completed",
      "attention_dtype": "float16",
      "device": str(extractor.device),
      "cuda_device": (torch.cuda.get_device_name(extractor.device)
                      if str(extractor.device).startswith("cuda") else None),
      "versions": {
          "python": platform.python_version(),
          "numpy": np.__version__,
          "torch": torch.__version__,
          "transformers": transformers.__version__,
          "cuda": torch.version.cuda,
      },
      "shards": [],
  }

  bytes_written = 0
  for index, example in enumerate(examples):
    attention = extractor.get_attn_maps([example])[0]
    expected_layers = getattr(
        extractor.model.config, "n_layer", attention.shape[0])
    expected_heads = getattr(
        extractor.model.config, "n_head", attention.shape[1])
    if attention.shape[:2] != (expected_layers, expected_heads):
      raise ValueError(
          "attention layer/head shape does not match model config")
    row_sum_error, future_max = validate_attention(attention)
    attention_filename = "sample_{:06d}_attn.npy".format(index)
    metadata_filename = "sample_{:06d}.json".format(index)
    metadata = {
        "sample_index": example.features.get("sample_index", index),
        "input_ids": example.input_ids,
        "tokens": example.tokens,
        "sequence_length": len(example.input_ids),
        "attention_shape": list(attention.shape),
        "attention_file": attention_filename,
        "row_sum_max_error": row_sum_error,
        "future_attention_max": future_max,
        "features": {
            key: value for key, value in example.features.items()
            if key not in ("input_ids", "tokens", "attns")
        },
    }
    metadata_bytes = len(json.dumps(
        metadata, ensure_ascii=False, indent=2,
        sort_keys=True).encode("utf-8"))
    projected_bytes = (
        bytes_written + attention.nbytes + metadata_bytes + 256
        + MANIFEST_RESERVE_BYTES)
    if (max_output_bytes is not None
        and projected_bytes > max_output_bytes):
      if stop_on_budget:
        manifest["stop_reason"] = "storage_budget"
        break
      raise RuntimeError(
          "next attention shard would exceed --max-output-bytes")

    attention_path = os.path.join(output_dir, attention_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)
    np.save(attention_path, attention, allow_pickle=False)
    write_json(metadata, metadata_path)
    attention_file_bytes = os.path.getsize(attention_path)
    metadata_file_bytes = os.path.getsize(metadata_path)
    bytes_written += attention_file_bytes + metadata_file_bytes
    manifest["shards"].append({
        "sample_index": metadata["sample_index"],
        "sequence_length": metadata["sequence_length"],
        "attention_shape": metadata["attention_shape"],
        "attention_file": attention_filename,
        "metadata_file": metadata_filename,
        "attention_bytes": attention_file_bytes,
        "metadata_bytes": metadata_file_bytes,
    })
    print("  shard {:}: length={}, total={:.2f} MiB".format(
        len(manifest["shards"]), len(example.input_ids),
        bytes_written / (1024.0 ** 2)))

  manifest["processed_count"] = len(manifest["shards"])
  manifest["unprocessed_count"] = (
      candidate_count - manifest["processed_count"])
  manifest["shard_bytes"] = bytes_written
  manifest_path = os.path.join(output_dir, "manifest.json")
  manifest["total_output_bytes"] = 0
  write_json(manifest, manifest_path)
  for _ in range(3):
    measured_size = directory_size(output_dir)
    if measured_size == manifest["total_output_bytes"]:
      break
    manifest["total_output_bytes"] = measured_size
    write_json(manifest, manifest_path)
  final_size = directory_size(output_dir)
  if max_output_bytes is not None and final_size > max_output_bytes:
    raise RuntimeError("output directory exceeded its storage budget")
  print("Wrote {} attention shards ({:.2f} MiB) to {}".format(
      manifest["processed_count"], final_size / (1024.0 ** 2), output_dir))
  if manifest["stop_reason"] == "storage_budget":
    print("Stopped before the next randomized candidate: storage budget")
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--preprocessed-data-file", required=True,
      help="Location of preprocessed JSON/JSONL data. Records may contain "
           "'input_ids', 'text', 'text_a'/'text_b', 'words', or 'tokens'.")
  parser.add_argument("--model-name", default="gpt2",
                      help="HuggingFace model name ('gpt2' == GPT-2 small).")
  parser.add_argument("--max_sequence_length", default=128, type=int,
                      help="Skip examples longer than this (default=128).")
  parser.add_argument("--batch_size", default=16, type=int,
                      help="Batch size when running GPT-2 (default=16).")
  parser.add_argument("--device", default=None,
                      help="torch device (default: cuda if available else cpu).")
  parser.add_argument(
      "--output-dir", default=None,
      help="Write validated .npy shards and manifest.json to this directory. "
           "When omitted, preserve the legacy single-pickle output.")
  parser.add_argument(
      "--max-output-bytes", default=None, type=int,
      help="Hard byte limit for sharded output, including its manifest.")
  parser.add_argument(
      "--stop-on-budget", action="store_true",
      help="Stop before the first shard that would exceed max-output-bytes.")
  args = parser.parse_args()

  if args.output_dir and args.batch_size != 1:
    raise ValueError("budgeted sharded extraction requires --batch_size 1")
  if args.stop_on_budget and not args.output_dir:
    raise ValueError("--stop-on-budget requires --output-dir")

  print("Building GPT-2 model...")
  extractor = AttnMapExtractor(args.model_name, args.device)

  print("Creating examples...")
  records = load_records(args.preprocessed_data_file)
  examples = []
  for features in records:
    example = Example(features, extractor.tokenizer)
    if len(example.input_ids) <= args.max_sequence_length:
      examples.append(example)

  if args.output_dir:
    print("Extracting sharded attention maps...")
    extract_sharded(
        extractor, examples, len(records), args.output_dir,
        max_output_bytes=args.max_output_bytes,
        stop_on_budget=args.stop_on_budget,
        max_sequence_length=args.max_sequence_length)
    print("Done!")
    return

  print("Extracting attention maps...")
  feature_dicts_with_attn = []
  n_batches = 1 + ((len(examples) - 1) // args.batch_size)
  for b, batch_of_examples in enumerate(
      examples_in_batches(examples, args.batch_size)):
    if b % 10 == 0 or b == n_batches - 1:
      print("  batch {:}/{:}".format(b + 1, n_batches))
    attns = extractor.get_attn_maps(batch_of_examples)
    for e, e_attn in zip(batch_of_examples, attns):
      e.features["attns"] = e_attn
      e.features["tokens"] = e.tokens
      feature_dicts_with_attn.append(e.features)

  outpath = args.preprocessed_data_file.replace(".json", "")
  outpath += "_attn.pkl"
  print("Writing attention maps to {:}...".format(outpath))
  write_pickle(feature_dicts_with_attn, outpath)
  print("Done!")


if __name__ == "__main__":
  main()
