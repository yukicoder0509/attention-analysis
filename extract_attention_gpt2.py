"""Runs GPT-2 over input data and writes out its attention maps to disk.

This is the GPT-2 counterpart of extract_attention.py (which runs BERT). It
produces pickle files in the *same* format expected by the analysis notebooks
and by head_distances.py:

    {
        "tokens": list of strings,
        "attns":  [n_layers, n_heads, n_tokens, n_tokens] float16 tensor
    }

Two things differ from the BERT extractor, both by design:

  * Special tokens. GPT-2 has no [CLS]/[SEP]. Following the port design, every
    BERT special token is substituted with GPT-2's document-boundary token
    <|endoftext|>. A single text segment becomes

        <|endoftext|> A <|endoftext|>

    and a pair of segments becomes

        <|endoftext|> A <|endoftext|> B <|endoftext|>

    The leading <|endoftext|> is the [CLS] analog (an attention "sink" that all
    later tokens can attend back to); an inter-segment <|endoftext|> is the
    [SEP] analog (a separator the second segment can attend back to). The
    trailing <|endoftext|> mirrors BERT's final [SEP] for structural parity,
    though under causal masking no earlier token can attend to it.

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
import pickle

import numpy as np
import torch
from transformers import GPT2Model, GPT2TokenizerFast


BOUNDARY_TOKEN = "<|endoftext|>"


def load_json(path):
  with open(path, "r") as f:
    return json.load(f)


def write_pickle(o, path):
  with open(path, "wb") as f:
    pickle.dump(o, f, -1)


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
        model_name, output_attentions=True)
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
        attention_mask=attention_mask.to(self.device))

    # tuple(len=n_layers) of [batch, n_heads, batch_max, batch_max]
    # -> [n_layers, batch, n_heads, batch_max, batch_max]
    attns = torch.stack(outputs.attentions, dim=0).cpu().numpy()

    results = []
    for i, seq_len in enumerate(seq_lens):
      # [n_layers, n_heads, seq_len, seq_len]; right-padding + causal mask means
      # the real tokens' rows/cols are unaffected by the padded positions.
      e_attn = attns[:, i, :, :seq_len, :seq_len].astype("float16")
      results.append(e_attn)
    return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--preprocessed-data-file", required=True,
      help="Location of preprocessed data (JSON file); a list of dicts with a "
           "'text', 'text_a'/'text_b', 'words', or 'tokens' field.")
  parser.add_argument("--model-name", default="gpt2",
                      help="HuggingFace model name ('gpt2' == GPT-2 small).")
  parser.add_argument("--max_sequence_length", default=128, type=int,
                      help="Skip examples longer than this (default=128).")
  parser.add_argument("--batch_size", default=16, type=int,
                      help="Batch size when running GPT-2 (default=16).")
  parser.add_argument("--device", default=None,
                      help="torch device (default: cuda if available else cpu).")
  args = parser.parse_args()

  print("Building GPT-2 model...")
  extractor = AttnMapExtractor(args.model_name, args.device)

  print("Creating examples...")
  examples = []
  for features in load_json(args.preprocessed_data_file):
    example = Example(features, extractor.tokenizer)
    if len(example.input_ids) <= args.max_sequence_length:
      examples.append(example)

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
