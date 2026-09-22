"""Visualization utilities for the AMLN framework.

Provides publication-quality plots for:
  - Latent space (t-SNE / PCA) with memory hills and slopes overlaid
  - Memory landscape height profile
  - Training loss curves
  - Attack chain graph visualization
  - Memory lifecycle (reinforcement/decay over time)
  - Threat evolution predictions
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from amln.data import ATTACK_FAMILIES
from amln.memory.graph import CyberMemoryGraph
from amln.memory.landscape import AdaptiveMemoryLandscape, RegionType


def _check_deps():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualization. "
                          "Install: pip install matplotlib")


def plot_latent_space(embeddings: np.ndarray, labels: np.ndarray,
                      landscape: AdaptiveMemoryLandscape,
                      method: str = "tsne",
                      title: str = "AMLN Latent Space",
                      save_path: Optional[str] = None) -> None:
    """Plot the 64-d latent space reduced to 2D with memory nodes overlaid.

    Args:
        embeddings: (N, latent_dim) event embeddings
        labels: (N,) integer labels
        landscape: populated landscape (hills/slopes drawn as markers)
        method: 'tsne' or 'pca'
        save_path: if provided, saves figure to this path
    """
    _check_deps()
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for dimensionality reduction.")

    # Combine event embeddings with node centroids for a joint projection
    centroids = landscape._matrix()
    all_vecs = np.vstack([embeddings, centroids]) if len(centroids) else embeddings

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_vecs) - 1))
    else:
        reducer = PCA(n_components=2, random_state=42)
    coords_2d = reducer.fit_transform(all_vecs)

    event_coords = coords_2d[:len(embeddings)]
    node_coords = coords_2d[len(embeddings):]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot events colored by label
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        name = ATTACK_FAMILIES[lab] if lab < len(ATTACK_FAMILIES) else f"family_{lab}"
        ax.scatter(event_coords[mask, 0], event_coords[mask, 1],
                   c=[cmap(i)], s=8, alpha=0.4, label=name)

    # Overlay memory nodes
    if len(node_coords):
        for i, node in enumerate(landscape.nodes):
            region = node.region(landscape.cfg)
            if region == RegionType.HILL:
                marker, size, edgecolor = "^", 120, "red"
            else:
                marker, size, edgecolor = "s", 60, "orange"
            ax.scatter(node_coords[i, 0], node_coords[i, 1],
                       marker=marker, s=size, edgecolors=edgecolor,
                       facecolors="none", linewidths=2, zorder=5)

    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_memory_heights(landscape: AdaptiveMemoryLandscape,
                        title: str = "Memory Landscape Height Profile",
                        save_path: Optional[str] = None) -> None:
    """Bar chart of memory node heights colored by region type."""
    _check_deps()
    if not landscape.nodes:
        print("  (no memory nodes to plot)")
        return

    heights = [n.height for n in landscape.nodes]
    regions = [n.region(landscape.cfg) for n in landscape.nodes]

    colors = {RegionType.HILL: "#2ecc71", RegionType.SLOPE: "#f39c12",
              RegionType.VALLEY: "#e74c3c"}
    bar_colors = [colors[r] for r in regions]

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(heights))
    ax.bar(x, heights, color=bar_colors, width=0.8)
    ax.axhline(landscape.cfg.hill_threshold, color="#2ecc71", ls="--",
               label=f"Hill threshold ({landscape.cfg.hill_threshold})")
    ax.axhline(landscape.cfg.valley_threshold, color="#e74c3c", ls="--",
               label=f"Valley threshold ({landscape.cfg.valley_threshold})")
    ax.set_xlabel("Memory Node Index")
    ax.set_ylabel("Height (stability)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_loss(losses: list[float],
                       title: str = "Autoencoder Pre-training Loss",
                       save_path: Optional[str] = None) -> None:
    """Plot reconstruction loss over epochs."""
    _check_deps()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, "b-o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_memory_graph(graph: CyberMemoryGraph, landscape: AdaptiveMemoryLandscape,
                      title: str = "Cyber Memory Graph",
                      save_path: Optional[str] = None) -> None:
    """Visualize the memory graph with nodes as circles and edges as arrows."""
    _check_deps()
    if not graph.edges:
        print("  (no graph edges to plot)")
        return

    # Layout via PCA of centroids (or random if few nodes)
    n_nodes = len(landscape.nodes)
    if n_nodes < 2:
        return

    mat = landscape._matrix()
    if HAS_SKLEARN:
        coords = PCA(n_components=2, random_state=42).fit_transform(mat)
    else:
        coords = mat[:, :2]  # fallback: first 2 dims

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges
    for (src, tgt), edge in graph.edges.items():
        if src >= n_nodes or tgt >= n_nodes:
            continue
        alpha = min(1.0, edge.weight / 3.0)
        ax.annotate("", xy=coords[tgt], xytext=coords[src],
                    arrowprops=dict(arrowstyle="->", color="gray",
                                    alpha=alpha, lw=0.5 + edge.weight))

    # Draw nodes
    for i, node in enumerate(landscape.nodes):
        region = node.region(landscape.cfg)
        color = "#2ecc71" if region == RegionType.HILL else "#f39c12"
        size = 40 + node.height * 20
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=size,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(str(node.label), (coords[i, 0], coords[i, 1]),
                    fontsize=7, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_continual_metrics(metrics_log: list[dict],
                           title: str = "Continual Learning Metrics",
                           save_path: Optional[str] = None) -> None:
    """Plot how memory metrics evolve over the continual learning stream."""
    _check_deps()
    if not metrics_log:
        return

    steps = [m.get("step", i) for i, m in enumerate(metrics_log)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Nodes over time
    ax = axes[0, 0]
    ax.plot(steps, [m.get("total_nodes", 0) for m in metrics_log], "b-")
    ax.set_ylabel("Total Nodes")
    ax.set_title("Memory Size")
    ax.grid(True, alpha=0.3)

    # Hills vs slopes
    ax = axes[0, 1]
    ax.plot(steps, [m.get("hills", 0) for m in metrics_log], "g-", label="Hills")
    ax.plot(steps, [m.get("slopes", 0) for m in metrics_log], "orange", label="Slopes")
    ax.legend()
    ax.set_ylabel("Count")
    ax.set_title("Hills vs Slopes")
    ax.grid(True, alpha=0.3)

    # Compression ratio
    ax = axes[1, 0]
    ax.plot(steps, [m.get("compression", 1.0) for m in metrics_log], "r-")
    ax.set_ylabel("Compression Ratio")
    ax.set_xlabel("Events Processed")
    ax.set_title("Memory Compression")
    ax.grid(True, alpha=0.3)

    # Forgetting efficiency
    ax = axes[1, 1]
    ax.plot(steps, [m.get("forgetting_eff", 0.0) for m in metrics_log], "m-")
    ax.set_ylabel("Forgetting Efficiency")
    ax.set_xlabel("Events Processed")
    ax.set_title("Selective Forgetting")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()
