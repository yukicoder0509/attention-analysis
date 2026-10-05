"""Prepare reproducible two-paragraph Wikipedia inputs for GPT-2 analysis.

The resulting JSONL records contain exact GPT-2 token ids for inputs of the
form

    <|endoftext|> paragraph_1 \n\n paragraph_2 <|endoftext|>

There is deliberately no document-boundary token between the paragraphs.
"""

import argparse
import hashlib
import json
import os
import random
import re

from transformers import GPT2TokenizerFast
from transformers.utils.hub import cached_file, extract_commit_hash


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
PARAGRAPH_SEPARATOR = "\n\n"


def resolve_hf_revision(model_name, filename):
  """Returns the commit encoded in a locally resolved HF snapshot path."""
  try:
    resolved_file = cached_file(
        model_name, filename, local_files_only=True)
  except (OSError, ValueError):
    return None
  return extract_commit_hash(resolved_file, None)


def split_paragraphs(text):
  """Returns non-empty paragraphs while preserving their document order."""
  return [paragraph.strip() for paragraph in PARAGRAPH_SPLIT_RE.split(text)
          if paragraph.strip()]


def truncate_around_boundary(ids_a, ids_b, token_budget):
  """Keeps the end of A and start of B within a balanced token budget."""
  if token_budget < 2:
    raise ValueError("token budget must leave room for both paragraphs")

  budget_a = token_budget // 2
  budget_b = token_budget - budget_a
  keep_a = min(len(ids_a), budget_a)
  keep_b = min(len(ids_b), budget_b)
  remaining = token_budget - keep_a - keep_b

  available_a = len(ids_a) - keep_a
  available_b = len(ids_b) - keep_b
  while remaining and (available_a or available_b):
    if available_a >= available_b and available_a:
      take = min(remaining, available_a)
      keep_a += take
      available_a -= take
    elif available_b:
      take = min(remaining, available_b)
      keep_b += take
      available_b -= take
    remaining -= take

  return ids_a[-keep_a:], ids_b[:keep_b]


def build_input(tokenizer, paragraph_a, paragraph_b, max_length):
  """Compatibility entry point; always tokenize the joined paragraph text."""
  return build_natural_input(tokenizer, paragraph_a, paragraph_b, max_length)


def build_natural_input(tokenizer, paragraph_a, paragraph_b, max_length=1024):
  """Tokenize the joined text, preserving context-dependent newline BPE.

  Offset mappings identify the natural boundary. In particular, encoding the
  separator independently would incorrectly force ID 628 in ordinary text.
  Truncation uses character boundaries then retokenizes to avoid partial UTF-8.
  """
  if not paragraph_a.strip() or not paragraph_b.strip():
    raise ValueError("both paragraphs must contain at least one token")
  original_a = tokenizer.encode(paragraph_a, add_special_tokens=False)
  original_b = tokenizer.encode(paragraph_b, add_special_tokens=False)
  left, right = paragraph_a, paragraph_b
  for _ in range(20):
    text = left + PARAGRAPH_SEPARATOR + right
    encoded = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True, verbose=False)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    boundary = [i for i, (a, b) in enumerate(offsets)
                if b > len(left) and a < len(left) + 2]
    if tokenizer.eos_token_id in ids:
      raise ValueError("paragraph text contains GPT-2's EOS token")
    if not boundary:
      raise ValueError("natural paragraph boundary was lost")
    if len(ids) + 2 <= max_length:
      break
    left_ids = list(range(boundary[0]))
    right_ids = list(range(boundary[-1] + 1, len(ids)))
    kept_left, kept_right = truncate_around_boundary(
        left_ids, right_ids, max_length - 2 - len(boundary) - 1)
    left = text[offsets[kept_left[0]][0]:len(left)]
    right = text[len(text) - len(right):offsets[kept_right[-1]][1]]
  else:
    raise ValueError("could not fit a canonical two-paragraph input")
  input_ids = [tokenizer.eos_token_id] + ids + [tokenizer.eos_token_id]
  return {
      "input_ids": input_ids,
      "tokens": tokenizer.convert_ids_to_tokens(input_ids),
      "sequence_length": len(input_ids),
      "boundary_positions": [i + 1 for i in boundary],
      "separator_token_ids": [ids[i] for i in boundary],
      "retained_text": text,
      "paragraph_a_token_count": boundary[0],
      "paragraph_b_token_count": len(ids) - boundary[-1] - 1,
      "paragraph_a_original_token_count": len(original_a),
      "paragraph_b_original_token_count": len(original_b),
      "tokenization": "joined_text_offsets_v2",
  }


def record_digest(record):
  """Returns a stable digest of the exact model input and its provenance."""
  digest_fields = {
      "article_id": record["article_id"],
      "paragraph_indices": record["paragraph_indices"],
      "input_ids": record["input_ids"],
  }
  payload = json.dumps(
      digest_fields, ensure_ascii=False, sort_keys=True,
      separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", default="wikimedia/wikipedia")
  parser.add_argument("--config", default="20231101.en")
  parser.add_argument("--revision", default="ad5752b")
  parser.add_argument("--model-name", default="openai-community/gpt2")
  parser.add_argument("--num-candidates", default=1000, type=int)
  parser.add_argument("--seed", default=42, type=int)
  parser.add_argument("--max-length", default=1024, type=int)
  parser.add_argument("--min-paragraph-tokens", default=32, type=int)
  parser.add_argument("--shuffle-buffer", default=10000, type=int)
  parser.add_argument("--output", required=True)
  parser.add_argument("--reprocess-jsonl", help="Retokenize existing article/paragraph choices without resampling")
  return parser.parse_args()


def main():
  args = parse_args()
  if args.reprocess_jsonl:
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.abspath(args.output) == os.path.abspath(args.reprocess_jsonl):
      raise ValueError("reprocessing requires a different output path")
    count = 0
    with open(args.reprocess_jsonl, encoding="utf-8") as source, open(
        args.output, "x", encoding="utf-8") as target:
      for line in source:
        record = json.loads(line)
        record["legacy_sha256"] = record.get("sha256")
        record.update(build_natural_input(tokenizer, record["paragraph_a"],
                                          record["paragraph_b"], args.max_length))
        record["sha256"] = record_digest(record)
        target.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
    print("Retokenized {} records to {}".format(count, args.output))
    return
  try:
    from datasets import load_dataset
  except ImportError as exc:
    raise SystemExit(
        "The 'datasets' package is required; install requirements-gpt2.txt") \
        from exc

  tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
  tokenizer_revision = (
      tokenizer.init_kwargs.get("_commit_hash")
      or resolve_hf_revision(args.model_name, "tokenizer_config.json"))
  dataset = load_dataset(
      args.dataset, args.config, revision=args.revision, split="train",
      streaming=True)
  dataset = dataset.shuffle(
      seed=args.seed, buffer_size=args.shuffle_buffer)
  rng = random.Random(args.seed)

  output_dir = os.path.dirname(os.path.abspath(args.output))
  os.makedirs(output_dir, exist_ok=True)
  count = 0
  with open(args.output, "w", encoding="utf-8") as output_file:
    for article in dataset:
      paragraphs = split_paragraphs(article.get("text", ""))
      encoded = [tokenizer.encode(p, add_special_tokens=False)
                 for p in paragraphs]
      valid_starts = [
          i for i in range(len(paragraphs) - 1)
          if len(encoded[i]) >= args.min_paragraph_tokens
          and len(encoded[i + 1]) >= args.min_paragraph_tokens
          and tokenizer.eos_token_id not in encoded[i]
          and tokenizer.eos_token_id not in encoded[i + 1]
      ]
      if not valid_starts:
        continue

      paragraph_index = rng.choice(valid_starts)
      try:
        model_input = build_natural_input(
            tokenizer, paragraphs[paragraph_index],
            paragraphs[paragraph_index + 1], args.max_length)
      except ValueError:
        continue

      record = {
          "sample_index": count,
          "article_id": str(article.get("id", "")),
          "title": article.get("title", ""),
          "url": article.get("url", ""),
          "dataset": args.dataset,
          "dataset_config": args.config,
          "dataset_revision": args.revision,
          "tokenizer_name": tokenizer.name_or_path,
          "tokenizer_revision": tokenizer_revision,
          "paragraph_indices": [paragraph_index, paragraph_index + 1],
          "paragraph_a": paragraphs[paragraph_index],
          "paragraph_b": paragraphs[paragraph_index + 1],
          "separator": PARAGRAPH_SEPARATOR,
          "seed": args.seed,
          "max_length": args.max_length,
          "sequence_length": len(model_input["input_ids"]),
      }
      record.update(model_input)
      record["sha256"] = record_digest(record)
      output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
      count += 1
      if count >= args.num_candidates:
        break

  if count != args.num_candidates:
    raise RuntimeError(
        "dataset ended after {:,} valid candidates; requested {:,}".format(
            count, args.num_candidates))
  print("Wrote {:,} candidates to {}".format(count, args.output))


if __name__ == "__main__":
  main()
