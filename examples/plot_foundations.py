"""Regenerate the course's explanatory plots with: python examples/plot_foundations.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#3455a4"
TEAL = "#087f8c"
ORANGE = "#d16b2f"
INK = "#17253b"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": INK,
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def frequency_plot() -> None:
    position = np.linspace(0, 40, 801)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True, layout="constrained")
    for ax, omega, color in zip(axes, (1.0, 0.1), (BLUE, TEAL)):
        ax.plot(position, np.sin(omega * position), color=color, lw=2.3, label=f"sin({omega:g} × position)")
        ax.plot(position, np.cos(omega * position), color=ORANGE, lw=1.7, alpha=0.85,
                label=f"cos({omega:g} × position)")
        ax.axhline(0, color="#bbc7d4", lw=0.8)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel("coordinate value")
        ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
        ax.text(0.01, 0.07, f"period = {2*np.pi/omega:.1f} tokens", transform=ax.transAxes,
                color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    axes[-1].set_xlabel("token position")
    fig.suptitle("One token step advances phase by ω radians", fontsize=15, weight="bold")
    save(fig, "frequency_scales")


def softmax_plot() -> None:
    logits = np.array([2.0, 1.0, 0.0])
    temperatures = (0.5, 1.0, 2.0)
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    for offset, temperature, color in zip((-0.25, 0, 0.25), temperatures, (BLUE, TEAL, ORANGE)):
        scaled = logits / temperature
        probabilities = np.exp(scaled - scaled.max())
        probabilities /= probabilities.sum()
        bars = ax.bar(x + offset, probabilities, 0.23, label=f"temperature {temperature:g}", color=color)
        for bar, value in zip(bars, probabilities):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.014, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, ("token A: 2", "token B: 1", "token C: 0"))
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("next-token probability")
    ax.set_title("Softmax temperature changes concentration, not logit order", fontsize=14, weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    save(fig, "softmax_temperature")


def gradient_plot() -> None:
    weights = np.linspace(0, 2, 401)
    loss = (2 * weights - 3) ** 2
    start = 1.0
    gradient = 2 * (2 * start - 3) * 2
    end = start - 0.1 * gradient
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    ax.plot(weights, loss, color=BLUE, lw=2.5, label="loss: (2w - 3)²")
    ax.plot(weights, 1 + gradient * (weights - start), color=ORANGE, ls="--", lw=1.7,
            label="tangent at w = 1; slope = -4")
    ax.scatter([start, end], [(2*start-3)**2, (2*end-3)**2], c=[ORANGE, TEAL], s=80, zorder=5)
    ax.annotate("start: w=1, loss=1", (start, 1), xytext=(0.17, 0.76),
                textcoords="axes fraction", arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.annotate("after step: w=1.4, loss=0.04", (end, 0.04), xytext=(0.55, 0.45),
                textcoords="axes fraction", arrowprops={"arrowstyle": "->", "color": TEAL})
    ax.set_xlim(0.15, 1.9)
    ax.set_ylim(-0.18, 5.1)
    ax.set_xlabel("parameter w")
    ax.set_ylabel("squared error")
    ax.set_title("A gradient step follows the local slope downhill", fontsize=14, weight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "gradient_step")


def embedding_plot() -> None:
    words = ("cat", "kitten", "dog", "puppy")
    one_hot = np.eye(4)
    distance = np.linalg.norm(one_hot[:, None, :] - one_hot[None, :, :], axis=-1)
    # Illustrative points chosen by hand; these are not trained embeddings.
    toy = np.array([[-1.0, 0.0], [-0.78, 0.25], [0.9, -0.05], [1.15, 0.2]])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), layout="constrained")
    image = axes[0].imshow(distance, cmap="Blues", vmin=0, vmax=1.6)
    axes[0].set_xticks(range(4), words, rotation=35)
    axes[0].set_yticks(range(4), words)
    axes[0].set_title("One-hot: every distinct pair equally far")
    for row in range(4):
        for col in range(4):
            axes[0].text(col, row, f"{distance[row, col]:.2f}", ha="center", va="center",
                         color="white" if row != col else INK, fontsize=9)
    fig.colorbar(image, ax=axes[0], fraction=0.046, label="Euclidean distance")
    axes[1].scatter(toy[:2, 0], toy[:2, 1], s=90, color=BLUE, label="cat family")
    axes[1].scatter(toy[2:, 0], toy[2:, 1], s=90, color=TEAL, label="dog family")
    for word, (x, y) in zip(words, toy):
        axes[1].annotate(word, (x, y), xytext=(5, 7), textcoords="offset points")
    axes[1].set_xlim(-1.4, 1.55)
    axes[1].set_ylim(-0.55, 0.65)
    axes[1].set_xlabel("illustrative feature 1")
    axes[1].set_ylabel("illustrative feature 2")
    axes[1].set_title("Toy dense vectors can express clusters")
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("An ID does not supply semantic geometry by itself", fontsize=14, weight="bold")
    save(fig, "embedding_geometry")


def attention_geometry_plot() -> None:
    words = ["bank", "river", "loan", "crowded", "near", "money"]
    base = np.array([
        [1.00, 0.20, 0.00],
        [0.20, 1.10, 0.70],
        [0.80, 1.60, 0.30],
        [-0.30, 0.50, 1.40],
        [-0.90, 0.10, 0.80],
        [0.70, -0.80, 1.00],
    ])
    q = base[0] + np.array([0.15, 0.35, 0.25])
    keys = base + np.array([0.05, 0.10, 0.12])
    values = base + 0.55
    scores = keys @ q / np.sqrt(3.0)
    weights = np.exp(scores - scores.max())
    weights = weights / weights.sum()
    output = weights @ values

    fig = plt.figure(figsize=(11.5, 6.0), layout="constrained")

    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.scatter(base[:, 0], base[:, 1], base[:, 2], s=90, c=range(len(words)), cmap="viridis", alpha=0.75)
    ax3d.scatter(q[0], q[1], q[2], s=150, color="crimson", depthshade=True, label="query")
    ax3d.scatter(output[0], output[1], output[2], s=150, color="gold", depthshade=True, label="attention output")
    for idx in range(len(words)):
        ax3d.plot(
            [q[0], keys[idx, 0]],
            [q[1], keys[idx, 1]],
            [q[2], keys[idx, 2]],
            color="gray",
            alpha=0.35,
            lw=0.9,
        )
        ax3d.text(base[idx, 0], base[idx, 1], base[idx, 2] + 0.12, words[idx], fontsize=9)
    ax3d.set_title("Word embeddings in a 3D projection")
    ax3d.set_xlabel("dim 1")
    ax3d.set_ylabel("dim 2")
    ax3d.set_zlabel("dim 3")
    ax3d.legend(frameon=False, fontsize=9)

    axbar = fig.add_subplot(122)
    bars = axbar.bar(words, weights, color=[BLUE, TEAL, ORANGE, BLUE, TEAL, ORANGE])
    axbar.axhline(0, color=INK, lw=0.8)
    axbar.set_title("Softmax weights for the bank query row")
    axbar.set_ylabel("attention weight")
    axbar.set_ylim(0, max(weights) + 0.1)
    for bar, weight in zip(bars, weights):
        axbar.text(bar.get_x() + bar.get_width() / 2, weight + 0.02, f"{weight:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("A few words, projected into 3D, before and after attention", fontsize=14, weight="bold")
    save(fig, "attention_geometry")


def attention_plot() -> None:
    q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    k = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    scores = q @ k.T / np.sqrt(2.0)
    allowed = np.tril(np.ones((3, 3), dtype=bool))
    masked = np.where(allowed, scores, -np.inf)
    stable = masked - np.max(masked, axis=1, keepdims=True)
    exponentials = np.exp(stable)
    weights = exponentials / exponentials.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for ax, values, title, vmax in zip(
        axes,
        (scores, weights),
        ("Scaled QKᵀ scores", "After causal mask + row softmax"),
        (1.5, 1.0),
    ):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(3), ("key 0", "key 1", "key 2"))
        ax.set_yticks(range(3), ("query 0", "query 1", "query 2"))
        ax.set_title(title)
        for row in range(3):
            for col in range(3):
                label = "×" if values is weights and not allowed[row, col] else f"{values[row, col]:.2f}"
                color = "white" if values is weights and values[row, col] > 0.72 else INK
                ax.text(col, row, label, ha="center", va="center", color=color)
    fig.suptitle("Mask future keys before normalizing each query row", fontsize=14, weight="bold")
    save(fig, "attention_mask")


if __name__ == "__main__":
    frequency_plot()
    softmax_plot()
    gradient_plot()
    embedding_plot()
    attention_geometry_plot()
    attention_plot()
    print(f"Wrote plots to {OUT}")
