import os

import numpy as np

from extract_attention import sample_gpt2_attention

# Number of samples to analyze
n = int(os.environ.get("N_SAMPLES", 1000))

attention_map = sample_gpt2_attention(n_samples=n, seed=42)

# Experiment 1
# Rows of each [n_tokens, n_tokens] map sum to 1 over keys; the diagonal (self)
# and the first sub-diagonal (previous token) are just two entries of each row.
# Every token counts equally: sum over all tokens of all samples, then divide
# by the total token count (longer samples contribute more tokens).
attn_self = np.zeros((12, 12))  # [layer, head] average attention a token puts on itself
attn_prev = np.zeros((12, 12))  # [layer, head] average attention a token puts on the previous token
n_self_tokens = 0
n_prev_tokens = 0  # the first token of each sample has no previous token
for sample_attn in attention_map:
    attns = sample_attn["attns"].astype(np.float64)  # float16 sums lose precision
    n_tokens = attns.shape[-1]
    attn_self += np.diagonal(attns, axis1=2, axis2=3).sum(-1)
    attn_prev += np.diagonal(attns, offset=-1, axis1=2, axis2=3).sum(-1)
    n_self_tokens += n_tokens
    n_prev_tokens += n_tokens - 1

attn_self /= n_self_tokens
attn_prev /= max(1, n_prev_tokens)
print("Averaged over {:} tokens from {:} samples".format(n_self_tokens, n))

print("Attention on self:", attn_self)
print("Attention on previous token:", attn_prev)

# Heat maps: one panel per measure, shared color scale so the two are comparable
import matplotlib
matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)
np.savez(os.path.join(FIG_DIR, "self_prev_attention_n{:}.npz".format(n)),
         attn_self=attn_self, attn_prev=attn_prev, n_samples=n,
         n_tokens=n_self_tokens)

# Single-hue sequential ramp: light = little attention, dark = a lot
blues = LinearSegmentedColormap.from_list("blues", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
vmax = max(attn_self.max(), attn_prev.max())

fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for ax, data, title in [(axes[0], attn_self, "Attention to the same token"),
                        (axes[1], attn_prev, "Attention to the previous token")]:
    im = ax.imshow(data, cmap=blues, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=12, color="#222222")
    ax.set_xlabel("Head", color="#555555")
    ax.set_ylabel("Layer", color="#555555")
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.tick_params(colors="#555555", length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Label only the strongest head in each panel
    layer, head = np.unravel_index(data.argmax(), data.shape)
    ax.text(head, layer, "{:.2f}".format(data[layer, head]), ha="center",
            va="center", fontsize=7, color="white")

cbar = fig.colorbar(im, ax=axes, shrink=0.8)
cbar.set_label("Mean attention weight (rows sum to 1)", color="#555555")
cbar.outline.set_visible(False)
cbar.ax.tick_params(colors="#555555", length=0)
fig.suptitle("GPT-2 attention per head, averaged over {:,} tokens from {:} Wikipedia samples".format(
                 n_self_tokens, n),
             fontsize=13, color="#222222")

fig_path = os.path.join(FIG_DIR, "self_prev_attention_n{:}.png".format(n))
fig.savefig(fig_path, dpi=200, facecolor="white")
print("Saved heat map to", fig_path)
