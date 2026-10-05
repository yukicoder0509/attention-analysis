import json
import os
import pickle

import torch
from transformers import GPT2Model, GPT2Tokenizer

MAX_SEQ_LENGTH = 1024 # The GPT-2 pretraining sequence length
SHARD_SIZE = 10  # samples per pickle; up to ~300 MB each at 1024 tokens
OUT_DIR = "wikipedia_gpt2_attn"
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "wikipedia_samples.json")


def load_model(device=None):
    """Returns (tokenizer, model) with eager attention so attentions are returned."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
    model.to(device)
    model.eval()
    return tokenizer, model


def download_wikipedia(n_samples=1024, seed=42, buffer_size=10000,
                       path=SAMPLES_PATH):
    """Streams n_samples random English Wikipedia articles and saves them as JSON."""
    from datasets import load_dataset

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True
    )
    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    random_samples = []
    for item in shuffled_dataset.take(n_samples):
        random_samples.append({
            "id": item["id"],
            "title": item["title"],
            "text": item["text"]
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(random_samples, f)
    print("Wrote {:} samples to {:}".format(len(random_samples), path))


def load_samples(n_samples=None, path=SAMPLES_PATH):
    """Returns the first n_samples (all if None) articles saved by download_wikipedia."""
    with open(path) as f:
        samples = json.load(f)
    if n_samples is not None:
        if n_samples > len(samples):
            raise ValueError("Requested {:} samples but {:} has only {:}".format(
                n_samples, path, len(samples)))
        samples = samples[:n_samples]
    return samples


def iter_attention(samples, tokenizer, model, max_seq_length=MAX_SEQ_LENGTH):
    """Yields each sample dict with "tokens" and "attns" added, one at a time.

    "attns" is a float16 numpy array of shape
    [n_layers, n_heads, n_tokens, n_tokens], same layout as the BERT script.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        for sample in samples:
            inputs = tokenizer(sample["text"], return_tensors="pt",
                               truncation=True, max_length=max_seq_length)
            inputs = inputs.to(device)
            outputs = model(**inputs, output_attentions=True)

            attns = torch.stack(outputs.attentions)[:, 0].half().cpu().numpy()
            yield {
                **sample,
                "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()),
                "attns": attns,
            }


def write_shard(feature_dicts, shard_idx, out_dir=OUT_DIR):
    outpath = os.path.join(out_dir, "shard_{:05d}.pkl".format(shard_idx))
    print("Writing attention maps to {:}...".format(outpath))
    with open(outpath, "wb") as f:
        pickle.dump(feature_dicts, f, -1)


def extract_attention(samples, tokenizer, model, out_dir=OUT_DIR,
                      shard_size=SHARD_SIZE, max_seq_length=MAX_SEQ_LENGTH):
    """Writes attention maps for samples to out_dir as pickle shards."""
    os.makedirs(out_dir, exist_ok=True)
    feature_dicts_with_attn = []
    shard_idx = 0
    for features in iter_attention(samples, tokenizer, model, max_seq_length):
        feature_dicts_with_attn.append(features)
        if len(feature_dicts_with_attn) == shard_size:
            write_shard(feature_dicts_with_attn, shard_idx, out_dir)
            feature_dicts_with_attn = []
            shard_idx += 1

    if feature_dicts_with_attn:
        write_shard(feature_dicts_with_attn, shard_idx, out_dir)


def sample_gpt2_attention(n_samples=1000, out_dir=None, tokenizer=None, model=None,
                          shard_size=SHARD_SIZE, max_seq_length=MAX_SEQ_LENGTH,
                          samples_path=SAMPLES_PATH):
    """Extracts GPT-2 attention for the first n_samples saved Wikipedia articles.

    If out_dir is None, returns the list of feature dicts (all attention maps
    are kept in memory: up to ~300 MB per sample at 1024 tokens). Otherwise
    writes pickle shards to out_dir and returns None.
    """
    if tokenizer is None or model is None:
        tokenizer, model = load_model()
    random_samples = load_samples(n_samples, samples_path)
    print("Sample ids: ", [sample["id"] for sample in random_samples])

    if out_dir is None:
        return list(iter_attention(random_samples, tokenizer, model, max_seq_length))
    extract_attention(random_samples, tokenizer, model, out_dir=out_dir,
                      shard_size=shard_size, max_seq_length=max_seq_length)


if __name__ == "__main__":
    sample_gpt2_attention(1000, out_dir=OUT_DIR)
    print("Done!")
