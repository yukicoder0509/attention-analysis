"""Streams random Wikipedia articles once and saves them to gpt_2/data/."""
import os
import sys

from extract_attention import download_wikipedia

download_wikipedia(n_samples=4096, seed=42)

# Streaming leaves background threads alive that abort the interpreter at
# shutdown ("PyGILState_Release ... finalizing"); the JSON is already closed.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
