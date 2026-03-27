"""
Dimensionality-reduction visualizations for groove embeddings,
with both static (matplotlib) and interactive (plotly) output.

Any method registered in :mod:`reduce_dimensionality` can be used.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pathlib

from groovestim.ratings.reduce_dimensionality import reduce_to_n_dims, METHOD_LABELS
from groovestim.utils.io_utils import DEFAULT_RATINGS, _format_annotated_dataset_all_ratings, _get_song_names, _get_song_styles

# ── Colormap & style constants ──────────────────────────────────────────
CMAP = "viridis"
POINT_SIZE = 48
POINT_ALPHA = 0.7
FIGSIZE_SINGLE = (16, 3.5)     # one method, 4 ratings
FIGSIZE_COMPARISON = (16, 7)   # two methods, 4 ratings

# Matplotlib markers cycled per musical style
_MPL_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]
# Plotly marker symbols cycled per musical style
_PLOTLY_SYMBOLS = ["circle", "square", "diamond", "cross", "triangle-up",
                   "triangle-down", "star", "hexagon", "pentagon", "x"]


# ── Data preparation helpers ────────────────────────────────────────────

def _project_in_2D(formatted_embeddings, method, gt_ratings=None):
    """Return 2-D projected embeddings for the given method name.

    Calls ``reduce_to_n_dims()`` and keeps only the first two components.
    """
    projected = reduce_to_n_dims(formatted_embeddings, method, n_components=2, gt_ratings=gt_ratings)
    return projected[:, :2]


def _prepare_projection(embeddings, groove_ratings_dataframe, method, annotate=False):
    """
    Shared data-preparation pipeline for all visualisation functions.

    Returns
    -------
    projected : np.ndarray of shape (n_samples, 2)
    y_ratings : dict[str, np.ndarray]
    song_names : list[str] or None
    styles : list[str] or None
    method_label : str
    """
    formatted_embeddings, y_ratings = _format_annotated_dataset_all_ratings(
        embeddings, groove_ratings_dataframe,
    )
    projected = _project_in_2D(formatted_embeddings, method, gt_ratings=y_ratings['groove'])
    song_names = _get_song_names(embeddings, groove_ratings_dataframe) if annotate else None
    styles = _get_song_styles(embeddings, groove_ratings_dataframe)
    method_label = METHOD_LABELS.get(method, method)
    return projected, y_ratings, song_names, styles, method_label


# ── Matplotlib (static) ─────────────────────────────────────────────────

def save_viz(embeddings, groove_ratings_dataframe, method="umap",
              ratings=DEFAULT_RATINGS, save_path=None, annotate=False, title_text=None,
              show_title=True, show_xlabel=True, show_ylabel=True):
    """
    Compute a 2-D projection and produce a static matplotlib figure.

    Parameters
    ----------
    embeddings : dict
        Per-song embeddings (song_id → array).
    groove_ratings_dataframe : DataFrame
        Dataframe with groove ratings.
    method : str
        Any method registered in :mod:`reduce_dimensionality`.
    ratings : str or list[str]
        Rating column(s) to plot. A single string is accepted.
    save_path : str or None
        If given, save the figure (PNG) to this path.
    annotate : bool
        If True, label each point with the song name.
    title_text : str or None
        Custom title prefix; defaults to ``"{method} projection"``.
    show_title : bool
        Whether to display the per-panel title.
    show_xlabel : bool
        Whether to display the x-axis label.
    show_ylabel : bool
        Whether to display the y-axis label.
    """
    if isinstance(ratings, str):
        ratings = [ratings]

    projected, y_ratings, song_names, styles, method_label = _prepare_projection(
        embeddings, groove_ratings_dataframe, method, annotate=annotate,
    )

    # Build style → marker mapping
    unique_styles = sorted(set(styles))
    style_to_marker = {
        s: _MPL_MARKERS[i % len(_MPL_MARKERS)] for i, s in enumerate(unique_styles)
    }
    style_indices = np.array([unique_styles.index(s) for s in styles])

    figsize = (max(5, 4 * len(ratings)), FIGSIZE_SINGLE[1])
    fig, axs = plt.subplots(
        1, len(ratings), constrained_layout=True, figsize=figsize, squeeze=False,
    )

    if title_text is not None:
        base_title = title_text
    else:
        base_title = f"{method_label} projection"

    for idx, rating_name in enumerate(ratings):
        ax = axs[0, idx]
        title = f"{base_title} | Rating: {rating_name}"
        # Plot each style group with its own marker
        for style_name in unique_styles:
            mask = style_indices == unique_styles.index(style_name)
            sc = ax.scatter(
                projected[mask, 0], projected[mask, 1],
                c=y_ratings[rating_name][mask],
                cmap=CMAP,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.3,
                marker=style_to_marker[style_name],
                vmin=y_ratings[rating_name].min(),
                vmax=y_ratings[rating_name].max(),

            )
        if show_title:
            ax.set_title(title, fontsize=11, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel(f"{method_label} 1", fontsize=9)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if show_ylabel:
            ax.set_ylabel(f"{method_label} 2", fontsize=9)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(rating_name, fontsize=9)

        if annotate and song_names is not None:
            for i, name in enumerate(song_names):
                ax.annotate(
                    name, (projected[i, 0], projected[i, 1]),
                    fontsize=5, alpha=0.6,
                )

    # Add a shared style legend with uniform gray markers (so color isn't
    # confused with the colormap, which encodes the rating value).
    legend_handles = [
        Line2D([0], [0], marker=style_to_marker[s], color="none",
               markerfacecolor="gray", markeredgecolor="white",
               markersize=6, label=s)
        for s in unique_styles
    ]
    axs[0, 0].legend(
        handles=legend_handles, title="Style",
        fontsize=7, title_fontsize=8, loc="best",
    )

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)

# ── Interactive Plotly visualizations ────────────────────────────────────

def save_interactive(embeddings, groove_ratings_dataframe, method="umap",
                     ratings=DEFAULT_RATINGS, save_path=None, title_text=None):
    """
    Produce an interactive HTML scatter plot (via Plotly) with hover
    tooltips showing the song name and all rating values.

    Parameters
    ----------
    ratings : str or list[str]
        Rating column(s) to plot. A single string is accepted.
    save_path : str or None
        Where to write the ``.html`` file.
    """
    if isinstance(ratings, str):
        ratings = [ratings]

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    projected, y_ratings, song_names, styles, method_label = _prepare_projection(
        embeddings, groove_ratings_dataframe, method, annotate=True,
    )

    # Build style → plotly marker symbol mapping
    unique_styles = sorted(set(styles))
    style_to_symbol = {
        s: _PLOTLY_SYMBOLS[i % len(_PLOTLY_SYMBOLS)] for i, s in enumerate(unique_styles)
    }
    symbols = [style_to_symbol[s] for s in styles]

    # Build a shared hover text showing all ratings + style
    hover_texts = []
    for i, name in enumerate(song_names):
        parts = [f"<b>{name}</b>", f"style: {styles[i]}"]
        for r in ratings:
            parts.append(f"{r}: {y_ratings[r][i]:.2f}")
        hover_texts.append("<br>".join(parts))

    fig = make_subplots(
        rows=1, cols=len(ratings),
        subplot_titles=[r for r in ratings],
        horizontal_spacing=0.06,
    )

    for idx, rating_name in enumerate(ratings):
        # Plot each style separately so it gets its own legend entry
        for style_name in unique_styles:
            mask = [s == style_name for s in styles]
            mask_arr = np.array(mask)
            fig.add_trace(
                go.Scatter(
                    x=projected[mask_arr, 0],
                    y=projected[mask_arr, 1],
                    mode="markers",
                    marker=dict(
                        color=y_ratings[rating_name][mask_arr],
                        colorscale="Viridis",
                        size=6,
                        opacity=0.8,
                        symbol=style_to_symbol[style_name],
                        colorbar=dict(
                            title=rating_name,
                            len=0.9,
                            x=1.0 if idx == len(ratings) - 1 else None,
                        ) if idx == len(ratings) - 1 else None,
                        line=dict(width=0.3, color="white"),
                    ),
                    text=[t for t, m in zip(hover_texts, mask) if m],
                    hoverinfo="text",
                    name=style_name,
                    legendgroup=style_name,
                    showlegend=(idx == 0),
                ),
                row=1, col=idx + 1,
            )
        fig.update_xaxes(title_text=f"{method_label} 1", row=1, col=idx + 1)
        fig.update_yaxes(title_text=f"{method_label} 2", row=1, col=idx + 1)

    if title_text is not None:
        title = title_text
    else:
        title = f"{method_label} projection"

    fig.update_layout(
        height=400,
        width=350 * len(ratings),
        template="plotly_dark",
        title_text=title,
        margin=dict(t=60, b=40),
    )

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive figure saved to {save_path}")

    return fig


# ── Main entry point (demo) ─────────────────────────────────────────────

if __name__ == "__main__":
    from groovestim.utils.io_utils import load_common_data
    from groovestim.models.get_embeddings import audio_embeddings

    model = "OpenMuQ"
    checkpoint = "MuQ-MuLan-large"
    audio_folder_path, groove_ratings_dataframe = load_common_data()

    song_audio_embeddings = audio_embeddings(
        model,
        audio_folder_path,
        checkpoint=checkpoint,
        space_mert="all",
        verbose=True,
        only_load_embeddings=True,
    )

    # Static per-method plots
    for method in ["umap", "tsne", "pca", "ica", "lle", "truncated_svd", "kernel_pca", "isomap", "rsa_reduce"]:
        # Static matplotlib file
        save_viz(song_audio_embeddings, groove_ratings_dataframe, method=method, save_path=f"/Brain/private/a23marmo/projects/groove_study/viz/static/{method}/{model}_{checkpoint}.png", annotate=False)

        # Interactive HTML
        save_interactive(song_audio_embeddings, groove_ratings_dataframe, method=method, save_path=f"/Brain/private/a23marmo/projects/groove_study/viz/interactive/{method}/{model}_{checkpoint}.html")
