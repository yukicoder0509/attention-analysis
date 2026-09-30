# BERT Attention Analysis

This repository contains code for [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341).
It includes code for getting attention maps from BERT and writing them to disk, analyzing BERT's attention in general (sections 3 and 6 of the paper), and comparing its attention to dependency syntax (sections 4.2 and 5).
We will add the code for the coreference resolution analysis (section 4.3 of the paper) soon!

## Requirements
For extracting attention maps from text:
* [Tensorflow](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)

Additional requirements for the attention analysis:
* [Jupyter](https://jupyter.org/https://jupyter.org/)
* [MatplotLib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/index.html)
* [scikit-learn](https://scikit-learn.org/)

## Attention Analysis
`Syntax_Analysis.ipynb` and `General_Analysis.ipynb`
contain code for analyzing BERT's attention, including reproducing the figures and tables in the paper.

You can download the data needed to run the notebooks (including BERT attention maps on Wikipedia
and the Penn Treebank) from [here](https://drive.google.com/open?id=1DEIBQIl0Q0az5ZuLoy4_lYabIfLSKBg-). However, note that the Penn Treebank annotations are not
freely available, so the Penn Treebank data only includes dummy labels.
If you want to run the analysis on your own data, you can use the scripts described below to extract BERT attention maps.

## Extracting BERT Attention Maps
We provide a script for running BERT over text and writing the resulting
attention maps to disk.
The input data should be a [JSON](https://www.json.org/) file containing a
list of dicts, each one corresponding to a single example to be passed in
to BERT. Each dict must contain exactly one of the following fields:
* `"text"`: A string.
* `"words"`: A list of strings. Needed if you want word-level rather than
token-level attention.
* `"tokens"`: A list of strings corresponding to BERT wordpiece tokenization.

If the present field is "tokens," the script expects [CLS]/[SEP] tokens
to be already added; otherwise it adds these tokens to the
beginning/end of the text automatically.
Note that if an example is longer than `max_sequence_length` tokens
after BERT wordpiece tokenization, attention maps will not be extracted for it.
Attention extraction adds two additional fields to each dict:
* `"attns"`: A numpy array of size [num_layers, heads_per_layer, sequence_length,
sequence_length] containing attention weights.
* `"tokens"`: If `"tokens"` was not already provided for the example, the
BERT-wordpiece-tokenized text (list of strings).

Other fields already in the feature dicts will be preserved. For example
if each dict has a `tags` key containing POS tags, they will stay
in the data after attention extraction so they can be used when
analyzing the data.

Attention extraction is run with
```
python extract_attention.py --preprocessed_data_file <path-to-your-data> --bert_dir <directory-containing-BERT-model>
```
The following optional arguments can also be added:
* `--max_sequence_length`: Maximum input sequence length after tokenization (default is 128).
* `--batch_size`: Batch size when running BERT over examples (default is 16).
* `--debug`: Use a tiny BERT model for fast debugging.
* `--cased`: Do not lowercase the input text.
* `--word_level`: Compute word-level instead of token-level attention (see Section 4.1 of the paper).

The feature dicts with added attention maps (numpy arrays with shape [n_layers, n_heads_per_layer, n_tokens, n_tokens]) are written to `<path-to-your-data>_attn.pkl`


## Pre-processing Scripts
We include two pre-processing scripts for going from a raw data file to
JSON that can be supplied to ``attention_extractor.py``.

`preprocess_unlabeled.py` does BERT-pre-training-style preprocessing for unlabeled text
(i.e, taking two consecutive text spans, truncating them so they are at most
`max_sequence_length` tokens, and adding [CLS]/[SEP] tokens).
Each line of the input data file
should be one sentence. Documents should be separated by empty lines.
Example usage:
```
python preprocess_unlabeled.py --data-file $ATTN_DATA_DIR/unlabeled.txt --bert-dir $ATTN_DATA_DIR/uncased_L-12_H-768_A-12
```
will create the file `$ATTN_DATA_DIR/unlabeled.json` containing pre-processed data.
After pre-processing, you can run `extract_attention.py` to get attention maps, e.g.,
```
python extract_attention.py --preprocessed-data-file $ATTN_DATA_DIR/unlabeled.json --bert-dir $ATTN_DATA_DIR/uncased_L-12_H-768_A-12
```


`preprocess_depparse.py` pre-processes dependency parsing data.
Dependency parsing data should consist of two files `train.txt` and `dev.txt` under a common directory.
Each line in the files should contain a word followed by a space followed by <index_of_head>-<dependency_label>
(e.g., 0-root). Examples should be separated by empty lines. Example usage:
```
python preprocess_depparse.py --data-dir $ATTN_DATA_DIR/depparse
```

After pre-processing, you can run `extract_attention.py` to get attention maps, e.g.,
```
python extract_attention.py --preprocessed-data-file $ATTN_DATA_DIR/depparse/dev.json --bert-dir $ATTN_DATA_DIR/uncased_L-12_H-768_A-12 --word_level
```
## GPT-2 small port

`extract_attention_gpt2.py` runs GPT-2 small (via HuggingFace `transformers`)
instead of BERT and writes attention maps in the **same pickle format** as
`extract_attention.py`, so the downstream tools (`head_distances.py`, the
analysis notebooks) work with minimal changes.

Requirements for this path: [PyTorch](https://pytorch.org/) and
[transformers](https://github.com/huggingface/transformers) (see
`requirements-gpt2.txt`). It uses
`GPT2Model.from_pretrained(model_name, output_attentions=True)` under the hood.

Two things differ from BERT, by design:
* **Special tokens.** GPT-2 has no `[CLS]`/`[SEP]`; both are substituted with
  the document-boundary token `<|endoftext|>`. A single segment becomes
  `<|endoftext|> A <|endoftext|>` and a pair becomes
  `<|endoftext|> A <|endoftext|> B <|endoftext|>`. The leading boundary token is
  the `[CLS]` analog (an attention sink later tokens attend back to); an
  inter-segment boundary token is the `[SEP]` analog.
* **Causal masking.** GPT-2 cannot attend to future tokens (`attns[i,j]=0` for
  `j>i`). Analyses about the *next* token (Sec 3.1) and about a *trailing*
  separator (`other -> [SEP]`, `[SEP] -> [SEP]`, entropy-from-`[CLS]`) are
  therefore expected to be empty — read them as architectural facts, not
  results. Prev-token, self, punctuation, head-entropy, and the Section-6 MDS
  clustering remain meaningful.

Input is the same JSON list-of-dicts as the BERT extractor. Each dict may
contain `"text"`, `"words"`, `"tokens"`, or the two-segment form
`"text_a"`/`"text_b"` (which places an `<|endoftext|>` separator between the two
passages — the causally-valid `[SEP]` analog). GPT-2 byte-level BPE tokenization
is applied automatically; boundary tokens are inserted for you unless you pass
pre-tokenized `"tokens"`. Word-level attention (`--word_level`) is **not**
supported for GPT-2 (it is hard-wired to WordPiece + `[CLS]`/`[SEP]`).

Example usage:
```
python extract_attention_gpt2.py --preprocessed-data-file $ATTN_DATA_DIR/unlabeled.json --model-name gpt2
```
Optional arguments: `--model-name` (default `gpt2` = GPT-2 small),
`--max_sequence_length` (default 128), `--batch_size` (default 16), `--device`.
Output is written to `<path>_attn.pkl` with the same
`[n_layers, n_heads, n_tokens, n_tokens]` `"attns"` array (144 heads for GPT-2
small, matching BERT-base).

For the general analysis, use `General_Analysis_GPT2.ipynb`, a copy of
`General_Analysis.ipynb` with the special-token detection adapted to
`<|endoftext|>` and notes on which sections are causally degenerate. The syntax
notebook (Sections 4.2/5) is **not** ported: it depends on word-level attention
alignment, which does not transfer to GPT-2's tokenizer.

## Computing Distances Between Attention Heads
`head_distances.py` computes the average Jenson-Shannon divergence between the attention weights of all pairs of attention heads and writes the results to disk as a numpy array of shape [n_heads, n_heads]. These distances can be used to cluster BERT's attention heads (see Section 6 and Figure 6 of the paper; code for doing this clustering is in `General_Analysis.ipynb`). Example usage (requires that attention maps have already been extracted):
```
python head_distances.py --attn-data-file $ATTN_DATA_DIR/unlabeled_attn.pkl --outfile $ATTN_DATA_DIR/head_distances.pkl
```

## Citation
If you find the code or data helpful, please cite the original paper:

```
@inproceedings{clark2019what,
  title = {What Does BERT Look At? An Analysis of BERT's Attention},
  author = {Kevin Clark and Urvashi Khandelwal and Omer Levy and Christopher D. Manning},
  booktitle = {BlackBoxNLP@ACL},
  year = {2019}
}
```

## Contact
[Kevin Clark](https://cs.stanford.edu/~kevclark/) (@clarkkev).
