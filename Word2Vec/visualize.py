import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Builds visualisations from our trained Word2Vec embeddings.
# Three outputs:
#   1. vectors.tsv + metadata.tsv — uploadable to https://projector.tensorflow.org
#      for an interactive 3D embedding explorer (no extra code required).
#   2. embeddings_tsne.png — static 2D scatter (t-SNE) of the most frequent words,
#      suitable as a static blog figure.
#   3. embeddings_umap_3d.html — fully interactive 3D scatter (UMAP, hover/rotate/zoom).
#      Embed in a blog post with <iframe src="embeddings_umap_3d.html">.
#
# Why two algorithms:
#   - t-SNE produces tight, visually obvious 2D clusters — great for a static figure.
#   - UMAP produces tighter clusters in 3D than t-SNE does (t-SNE in 3D tends toward
#     a uniform sphere because it has more "room" to satisfy local-neighbour constraints).
#
# Requires sklearn, umap-learn, matplotlib, plotly. If missing:
#   pip install scikit-learn umap-learn matplotlib plotly

# Tweakables
TOP_N = 200
TSNE_PERPLEXITY = 30   # for 2D t-SNE; try 5-50, larger emphasises broader structure
UMAP_N_NEIGHBORS = 15  # for 3D UMAP; smaller = tighter local clusters, larger = global structure
UMAP_MIN_DIST = 0.1    # for 3D UMAP; smaller = denser clusters
RANDOM_SEED = 42

# Load the trained embeddings + vocabulary saved by Word2Vec.py
embeddings_in = np.load("embeddings_in.npy")
vocabulary = open("vocabulary.txt").read().splitlines()

# Step 1: Export TSV files for TensorFlow's Embedding Projector.
# Upload both files at https://projector.tensorflow.org to explore interactively.
np.savetxt("vectors.tsv", embeddings_in, delimiter="\t", fmt="%.6f")
with open("metadata.tsv", "w") as f:
    f.write("\n".join(vocabulary))
print(f"Exported vectors.tsv and metadata.tsv ({len(vocabulary)} rows each).")

# Top N most frequent words — both static and interactive views use this slice.
# Vocabulary is already sorted by frequency desc, so the first N rows = top N words.
top_embeddings = embeddings_in[:TOP_N]
top_words = vocabulary[:TOP_N]

# Step 2: Run t-SNE in 2D for the static figure.
print(f"Running 2D t-SNE on top {TOP_N} words...")
tsne_2d = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=RANDOM_SEED)
points_2d = tsne_2d.fit_transform(top_embeddings)

# Plot it as a labelled scatter (static PNG)
fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(points_2d[:, 0], points_2d[:, 1], s=10, alpha=0.4, color="steelblue")
for i, word in enumerate(top_words):
    ax.annotate(word, (points_2d[i, 0], points_2d[i, 1]), fontsize=8, alpha=0.85)
ax.set_title(f"t-SNE projection of top {TOP_N} most frequent words (Shakespeare Word2Vec)")
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()

png_path = "embeddings_tsne.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {png_path}")

# Step 3: Run UMAP in 3D for the interactive figure.
# UMAP gives tighter clusters in 3D than t-SNE — see file header for why.
print(f"Running 3D UMAP on top {TOP_N} words...")
umap_3d = UMAP(
    n_components=3,
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    random_state=RANDOM_SEED,
)
points_3d = umap_3d.fit_transform(top_embeddings)

# Build an interactive Plotly 3D scatter.
# Labels are shown on hover only — drawing 500 floating text labels in 3D makes a
# cluttered mess, but a hover tooltip per marker keeps it readable.
interactive_fig = go.Figure(go.Scatter3d(
    x=points_3d[:, 0],
    y=points_3d[:, 1],
    z=points_3d[:, 2],
    mode="markers",
    text=top_words,
    marker=dict(size=4, color="steelblue", opacity=0.75),
    hoverinfo="text",
))
interactive_fig.update_layout(
    title=f"3D UMAP projection of top {TOP_N} most frequent words (Shakespeare Word2Vec)",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    width=1200,
    height=800,
)

html_path = "embeddings_umap_3d.html"
interactive_fig.write_html(html_path)
print(f"Saved {html_path} (interactive 3D — open in a browser, drag to rotate)")
