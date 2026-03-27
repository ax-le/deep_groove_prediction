"""
Unified groove estimation experiment runner.

Runs experiments for midi_drums and source_separated modalities.
"""

import itertools
import os
import pathlib

import numpy as np
import pandas as pd

import groovestim.ratings.linear_probing as linear_probing
import groovestim.ratings.rsa as rsa
import groovestim.ratings.compute_viz as compute_viz

import groovestim.utils.default_path as default_path
import groovestim.utils.io_utils as io_utils

# Global variable to control verbosity 
verbose = False

from groovestim.utils.io_utils import DEFAULT_RATINGS

# ── Experiment configuration ────────────────────────────────────────────
EXPERIMENTS = [
    # {
    #     "modality": "midi_drums",
    #     "models": {
    #         "OpenMuQ": ["MuQ-MuLan-large"],
    #     },
    # },
    {
        "modality": "source_separated",
        "models": {
            "OpenMuQ": ["MuQ-MuLan-large"],
        },
    },
]

N_CROSS_VAL = 4
N_CLOSEST = 5
SEEDS = [0, 1, 2, 3, 4]
VIZ_RATING = "Q4_party"
INSTRUMENTS = ["bass", "drums", "other", "vocals", "drums_and_bass"] # ["no_bass", "no_drums", "no_other", "no_vocals"] 
INSTRUMENTS_NAMES = {
    "bass": "Bass",
    "drums": "Drums",
    "other": "Other",
    "vocals": "Vocals",
    "drums_and_bass": "Drums and Bass",
}

# ── Shared helpers ──────────────────────────────────────────────────────

def _results_to_row(per_seed_results, seeds, modality, model, checkpoint, chosen_instrument=None):
    """
    Merge results from multiple seeds into a single flat row.

    Each rating gets its own set of columns (e.g. ``R2_groove_seed0``,
    ``R2_groove_mean``, ``R2_groove_std``).

    Parameters
    ----------
    per_seed_results : list[dict]
        Length-``len(seeds)`` list; each element is the output of
        ``compute_ratings_scores`` for one seed.
    seeds : list[int]
        The seed values used (for column naming).

    Returns
    -------
    row : dict
        A single dict with per-seed and aggregate R² columns for every
        rating.
    """
    rating_names = list(per_seed_results[0].keys())
    row = {
        "modality": modality,
        "model": model,
        "checkpoint": checkpoint,
        "chosen_instrument": chosen_instrument,
    }
    for rating_name in rating_names:
        r2_vals = []
        for seed, seed_results in zip(seeds, per_seed_results):
            r2 = seed_results[rating_name]["R2"]
            row[f"R2_{rating_name}_seed{seed}"] = r2
            r2_vals.append(r2)
        row[f"R2_{rating_name}_mean"] = np.mean(r2_vals)
        row[f"R2_{rating_name}_std"] = np.std(r2_vals)
    return row


def _print_model_results(row):
    """Print R² mean ± std for every rating in a single result row."""
    label_parts = [row["modality"], row["model"], row["checkpoint"]]
    if row.get("chosen_instrument"):
        label_parts.append(row["chosen_instrument"])
    label = " | ".join(str(p) for p in label_parts)
    r2_parts = []
    for rating in DEFAULT_RATINGS:
        mean_col = f"R2_{rating}_mean"
        std_col = f"R2_{rating}_std"
        if mean_col in row:
            r2_parts.append(f"{rating}: {row[mean_col]:.4f}±{row[std_col]:.4f}")
    print(f"  {label}:  R² = {', '.join(r2_parts)}")


def print_aggregated_results(results_df):
    """Print a summary table of mean ± std R² per condition."""
    print(f"\n{'='*80}")
    print("Aggregated results (mean ± std over seeds)")
    print(f"{'='*80}")
    for _, row in results_df.iterrows():
        _print_model_results(row.to_dict())

def load_embeddings(modality, chosen_instrument="all", model="OpenMuQ", checkpoint="MuQ-MuLan-large"):
    all_embeddings = {}
    match modality:
        case "midi_drums":
            midi_drums_folder_path = pathlib.Path(default_path.cache_dir_save_embeddings) / "midi_drums" / model / checkpoint

            for song_id in io_utils.get_song_ids():
                embedding_file = midi_drums_folder_path / f'audio_embedding_{song_id}.npz'
                if embedding_file.exists():
                    this_embedding = np.load(embedding_file, allow_pickle=True)
                    assert this_embedding['song_id'] == song_id
                    this_audio_embedding = this_embedding['audio_embedding']
                    if verbose:
                        print(f"  ✓ Loaded audio embedding for song {song_id} from cache")
                else:
                    raise FileNotFoundError(f"Embedding file not found for song {song_id}")

                all_embeddings[song_id] = this_audio_embedding

            return all_embeddings

        case "source_separated":
            source_separated_folder_path = pathlib.Path(default_path.cache_dir_save_embeddings) / "source_separated" / model / checkpoint
            for song_id in io_utils.get_song_ids():
                this_song_folder = source_separated_folder_path / song_id
                assert chosen_instrument in ["bass", "drums", "other", "vocals","no_bass", "no_drums", "no_other", "no_vocals", "drums_and_bass"]
                embedding_file = this_song_folder / f'audio_embedding_{chosen_instrument}.npz'
                if embedding_file.exists():
                    this_embedding = np.load(embedding_file, allow_pickle=True)
                    assert this_embedding['song_id'] == song_id
                    assert this_embedding['instrument_name'] == chosen_instrument
                    this_audio_embedding = this_embedding['audio_embedding']
                    if verbose:
                        print(f"  ✓ Loaded audio embedding for song {song_id} from cache")
                else:
                    raise FileNotFoundError(f"Embedding file not found for song {song_id}")

                all_embeddings[song_id] = this_audio_embedding

            return all_embeddings



# ── Per-modality runners ────────────────────────────────────────────────

def _prepare_projection_all_instruments(embeddings_dict, ratings_df, ratings, reduction_method, annotate=False):
    """
    Shared data-preparation pipeline for all-instruments visualisations.

    Returns
    -------
    projected : np.ndarray of shape (n_samples, 2)
    combined_y_ratings : dict[str, np.ndarray]
    instruments : np.ndarray of shape (n_samples,)
    method_label : str
    song_names : list[str] or None  (None when annotate=False)
    """
    from groovestim.ratings.reduce_dimensionality import reduce_to_n_dims, METHOD_LABELS
    from groovestim.utils.io_utils import _format_annotated_dataset_all_ratings, _get_song_names

    all_formatted = []
    all_y_ratings = {r: [] for r in ratings}
    instrument_labels = []
    song_names_list = [] if annotate else None

    for instrument, embeddings in embeddings_dict.items():
        formatted, y_ratings = _format_annotated_dataset_all_ratings(embeddings, ratings_df)
        all_formatted.append(formatted)
        instrument_labels.extend([instrument] * len(formatted))
        for r in ratings:
            all_y_ratings[r].append(y_ratings[r])
        if annotate:
            song_names_list.extend(_get_song_names(embeddings, ratings_df))

    combined_formatted = np.concatenate(all_formatted, axis=0)
    combined_y_ratings = {r: np.concatenate(all_y_ratings[r], axis=0) for r in ratings}

    gt_ratings = combined_y_ratings.get(VIZ_RATING)
    projected = reduce_to_n_dims(combined_formatted, reduction_method, n_components=2, gt_ratings=gt_ratings)
    projected = projected[:, :2]

    method_label = METHOD_LABELS.get(reduction_method, reduction_method)
    return projected, combined_y_ratings, np.array(instrument_labels), method_label, song_names_list


def plot_all_instruments_together(all_embeddings_dict, groove_ratings_dataframe,
                                  method="umap", ratings=DEFAULT_RATINGS,
                                  save_path=None, annotate=False, title_text=None):
    """
    Plot embeddings of all instruments in a single space, using marker shape
    to encode the instrument and colour to encode the rating.

    Parameters
    ----------
    all_embeddings_dict : dict[str, dict]
        instrument → per-song embedding dict.
    groove_ratings_dataframe : DataFrame
    method : str
    ratings : str or list[str]
        Rating column(s) to plot. A single string is accepted.
    save_path : str or None
    annotate : bool
        If True, label each point with the song name.
    title_text : str or None
        Base title; defaults to ``"<method> projection (All Instruments)"``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if isinstance(ratings, str):
        ratings = [ratings]

    from groovestim.ratings.compute_viz import (
        CMAP, POINT_SIZE, POINT_ALPHA, FIGSIZE_SINGLE, _MPL_MARKERS,
    )

    projected, y_ratings, instruments, method_label, song_names = \
        _prepare_projection_all_instruments(
            all_embeddings_dict, groove_ratings_dataframe, ratings, method, annotate=annotate,
        )

    unique_instruments = sorted(set(instruments))
    instrument_to_marker = {
        inst: _MPL_MARKERS[i % len(_MPL_MARKERS)] for i, inst in enumerate(unique_instruments)
    }

    base_title = title_text if title_text is not None else f"{method_label} projection (All Instruments)"
    figsize = (max(5, 4 * len(ratings)), FIGSIZE_SINGLE[1])
    fig, axs = plt.subplots(
        1, len(ratings), constrained_layout=True, figsize=figsize, squeeze=False,
    )

    for idx, rating_name in enumerate(ratings):
        ax = axs[0, idx]
        title = f"{base_title} | Rating: {rating_name}" if len(ratings) > 1 else base_title
        for instrument_name in unique_instruments:
            mask = instruments == instrument_name
            sc = ax.scatter(
                projected[mask, 0], projected[mask, 1],
                c=y_ratings[rating_name][mask],
                cmap=CMAP,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.3,
                marker=instrument_to_marker[instrument_name],
                vmin=y_ratings[rating_name].min(),
                vmax=y_ratings[rating_name].max(),
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{method_label} 1", fontsize=9)
        ax.set_ylabel(f"{method_label} 2", fontsize=9)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(rating_name, fontsize=9)

        if annotate and song_names is not None:
            for i, name in enumerate(song_names):
                ax.annotate(name, (projected[i, 0], projected[i, 1]), fontsize=5, alpha=0.6)

    # Grey legend so marker shape encodes instrument without misleading colors
    legend_handles = [
        Line2D([0], [0], marker=instrument_to_marker[inst], color="none",
               markerfacecolor="gray", markeredgecolor="white",
               markersize=6, label=inst)
        for inst in unique_instruments
    ]
    axs[0, 0].legend(handles=legend_handles, title="Instrument", fontsize=7, title_fontsize=8, loc="best")

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure with all instruments saved to {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)


def plot_four_instruments_2x2(all_embeddings_dict, groove_ratings_dataframe,
                               method="umap", rating="groove",
                               save_path=None, title_text=None):
    """
    Produce a 2×2 static matplotlib figure showing the embedding projection
    for the first four instruments (bass, drums, other, vocals), one per panel.

    Each instrument is projected independently into 2-D.  Colour encodes the
    requested *rating* (viridis colormap, one colorbar per panel).  Marker
    shape encodes the musical style, with a single shared legend in the
    top-left panel.  Axis labels follow the 2×2 grid convention:

        bass  (top-left)   → y-label ✓, x-label ✗
        drums (top-right)  → y-label ✗, x-label ✗
        other (bot-left)   → y-label ✓, x-label ✓
        vocals(bot-right)  → y-label ✗, x-label ✓

    Parameters
    ----------
    all_embeddings_dict : dict[str, dict]
        instrument → per-song embedding dict.  Must contain at least the
        four keys: ``\"bass\"``, ``\"drums\"``, ``\"other\"``, ``\"vocals\"``.
    groove_ratings_dataframe : DataFrame
    method : str
        Dimensionality-reduction method (registered in reduce_dimensionality).
    rating : str
        Rating column to use for the colour encoding.
    save_path : str or None
    title_text : str or None
        Suptitle; if None no suptitle is added.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from groovestim.ratings.reduce_dimensionality import reduce_to_n_dims, METHOD_LABELS
    from groovestim.utils.io_utils import (
        _format_annotated_dataset_all_ratings, _get_song_styles,
    )
    from groovestim.ratings.compute_viz import CMAP, POINT_SIZE, POINT_ALPHA, _MPL_MARKERS

    GRID_INSTRUMENTS = ["bass", "drums", "other", "vocals"]
    # (row, col) position in the 2×2 grid
    GRID_POSITIONS = {
        "bass":   (0, 0),
        "drums":  (0, 1),
        "other":  (1, 0),
        "vocals": (1, 1),
    }
    # Axis-label visibility rules
    SHOW_XLABEL = {"bass": False, "drums": False, "other": True, "vocals": True}
    SHOW_YLABEL = {"bass": True,  "drums": False, "other": True, "vocals": False}

    method_label = METHOD_LABELS.get(method, method)

    # GridSpec: 2 rows × 3 cols — col 2 is a sliver for the shared colorbar
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(11, 9))
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1, 1, 0.06],
        hspace=0.08,
        wspace=0.08,
    )
    ax_grid = {
        "bass":   fig.add_subplot(gs[0, 0]),
        "drums":  fig.add_subplot(gs[0, 1]),
        "other":  fig.add_subplot(gs[1, 0]),
        "vocals": fig.add_subplot(gs[1, 1]),
    }
    cbar_ax = fig.add_subplot(gs[:, 2])   # spans both rows

    # Global colour range (shared across all panels)
    global_y = []
    for instrument_name in GRID_INSTRUMENTS:
        if instrument_name not in all_embeddings_dict:
            continue
        _, y_all = _format_annotated_dataset_all_ratings(
            all_embeddings_dict[instrument_name], groove_ratings_dataframe,
        )
        global_y.append(y_all[rating])
    vmin = min(y.min() for y in global_y)
    vmax = max(y.max() for y in global_y)

    # Collect all styles across every grid instrument so the legend is complete
    all_styles: list[str] = []
    for instrument_name in GRID_INSTRUMENTS:
        if instrument_name not in all_embeddings_dict:
            continue
        all_styles.extend(
            _get_song_styles(all_embeddings_dict[instrument_name], groove_ratings_dataframe)
        )
    unique_styles = sorted(set(all_styles))
    style_to_marker = {
        s: _MPL_MARKERS[i % len(_MPL_MARKERS)] for i, s in enumerate(unique_styles)
    }

    # Legend handles (same for every panel)
    legend_handles = [
        Line2D([0], [0], marker=style_to_marker[s], color="none",
               markerfacecolor="gray", markeredgecolor="white",
               markersize=6, label=s)
        for s in unique_styles
    ]

    sc_last = None
    for instrument_name in GRID_INSTRUMENTS:
        if instrument_name not in all_embeddings_dict:
            continue

        embeddings = all_embeddings_dict[instrument_name]
        ax = ax_grid[instrument_name]

        # Project independently for this instrument
        formatted, y_ratings_all = _format_annotated_dataset_all_ratings(
            embeddings, groove_ratings_dataframe,
        )
        if rating not in y_ratings_all:
            raise ValueError(
                f"Unknown rating '{rating}'. Available ratings: {list(y_ratings_all.keys())}"
            )
        projected = reduce_to_n_dims(
            formatted, method, n_components=2,
            gt_ratings=y_ratings_all.get(rating),
        )[:, :2]
        y_rating = y_ratings_all[rating]

        styles = _get_song_styles(embeddings, groove_ratings_dataframe)
        styles_arr = np.array(styles)

        for style_name in unique_styles:
            mask = styles_arr == style_name
            sc = ax.scatter(
                projected[mask, 0], projected[mask, 1],
                c=y_rating[mask],
                cmap=CMAP,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.3,
                marker=style_to_marker[style_name],
                vmin=vmin,
                vmax=vmax,
            )
        sc_last = sc

        ax.set_aspect("equal", adjustable="datalim")

        # Panel title = instrument name
        ax.set_title(INSTRUMENTS_NAMES.get(instrument_name, instrument_name),
                     fontsize=10, fontweight="bold", pad=6)

        # Axis labels according to grid position
        if SHOW_XLABEL[instrument_name]:
            ax.set_xlabel(f"{method_label} 1", fontsize=9)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if SHOW_YLABEL[instrument_name]:
            ax.set_ylabel(f"{method_label} 2", fontsize=9)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        # Style legend on every panel
        ax.legend(
            handles=legend_handles, title="Style",
            fontsize=7, title_fontsize=8, loc="best",
        )

    # Single shared colorbar in the dedicated axes
    if sc_last is not None:
        cbar = fig.colorbar(sc_last, cax=cbar_ax)
        cbar.set_label(rating, fontsize=9)

    if title_text is not None:
        fig.suptitle(title_text, fontsize=12, fontweight="bold", y=1.01)

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"2×2 instrument figure saved to {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)


def plot_five_instruments_3rows(all_embeddings_dict, groove_ratings_dataframe,
                                method="umap", rating="groove",
                                save_path=None, title_text=None):
    """
    Produce a static matplotlib figure with:
      - row 1: bass (left), drums (right)
      - row 2: other (left), vocals (right)
      - row 3: drums_and_bass (single centered panel, same size as others)

    Each instrument is projected independently into 2-D. Colour encodes the
    requested *rating*. Marker shape encodes musical style.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from groovestim.ratings.reduce_dimensionality import reduce_to_n_dims, METHOD_LABELS
    from groovestim.utils.io_utils import (
        _format_annotated_dataset_all_ratings, _get_song_styles,
    )
    from groovestim.ratings.compute_viz import CMAP, POINT_SIZE, POINT_ALPHA, _MPL_MARKERS

    GRID_INSTRUMENTS = ["bass", "drums", "other", "vocals", "drums_and_bass"]
    TOP_GRID_INSTRUMENTS = ["bass", "drums", "other", "vocals"]
    SHOW_XLABEL = {
        "bass": False,
        "drums": False,
        "other": True,
        "vocals": True,
        "drums_and_bass": True,
    }
    SHOW_YLABEL = {
        "bass": True,
        "drums": False,
        "other": True,
        "vocals": False,
        "drums_and_bass": True,
    }

    panel_titles = {
        **INSTRUMENTS_NAMES,
        "drums_and_bass": "drums and bass",
    }

    method_label = METHOD_LABELS.get(method, method)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(11, 13))
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        width_ratios=[1, 1, 0.06],
        hspace=0.12,
        wspace=0.08,
    )
    ax_grid = {
        "bass": fig.add_subplot(gs[0, 0]),
        "drums": fig.add_subplot(gs[0, 1]),
        "other": fig.add_subplot(gs[1, 0]),
        "vocals": fig.add_subplot(gs[1, 1]),
    }

    # Create a centered panel in row 3 with same width as top panels.
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=gs[2, :2],
        width_ratios=[0.5, 1, 0.5],
        wspace=0.0,
    )
    ax_grid["drums_and_bass"] = fig.add_subplot(gs_bottom[0, 1])
    # Nudge the bottom subplot down a bit to avoid overlap with row 2.
    bottom_pos = ax_grid["drums_and_bass"].get_position()
    ax_grid["drums_and_bass"].set_position([
        bottom_pos.x0,
        bottom_pos.y0 - 0.02,
        bottom_pos.width,
        bottom_pos.height,
    ])
    cbar_ax = fig.add_subplot(gs[:, 2])

    global_y = []
    for instrument_name in GRID_INSTRUMENTS:
        if instrument_name not in all_embeddings_dict:
            continue
        _, y_all = _format_annotated_dataset_all_ratings(
            all_embeddings_dict[instrument_name], groove_ratings_dataframe,
        )
        if rating not in y_all:
            raise ValueError(
                f"Unknown rating '{rating}'. Available ratings: {list(y_all.keys())}"
            )
        global_y.append(y_all[rating])
    vmin = min(y.min() for y in global_y)
    vmax = max(y.max() for y in global_y)

    all_styles: list[str] = []
    for instrument_name in GRID_INSTRUMENTS:
        if instrument_name not in all_embeddings_dict:
            continue
        all_styles.extend(
            _get_song_styles(all_embeddings_dict[instrument_name], groove_ratings_dataframe)
        )
    unique_styles = sorted(set(all_styles))
    style_to_marker = {
        s: _MPL_MARKERS[i % len(_MPL_MARKERS)] for i, s in enumerate(unique_styles)
    }
    legend_handles = [
        Line2D([0], [0], marker=style_to_marker[s], color="none",
               markerfacecolor="gray", markeredgecolor="white",
               markersize=6, label=s)
        for s in unique_styles
    ]

    sc_last = None
    for instrument_name in TOP_GRID_INSTRUMENTS + ["drums_and_bass"]:
        if instrument_name not in all_embeddings_dict:
            continue

        embeddings = all_embeddings_dict[instrument_name]
        ax = ax_grid[instrument_name]

        formatted, y_ratings_all = _format_annotated_dataset_all_ratings(
            embeddings, groove_ratings_dataframe,
        )
        if rating not in y_ratings_all:
            raise ValueError(
                f"Unknown rating '{rating}'. Available ratings: {list(y_ratings_all.keys())}"
            )

        projected = reduce_to_n_dims(
            formatted, method, n_components=2,
            gt_ratings=y_ratings_all.get(rating),
        )[:, :2]
        y_rating = y_ratings_all[rating]

        styles = _get_song_styles(embeddings, groove_ratings_dataframe)
        styles_arr = np.array(styles)

        for style_name in unique_styles:
            mask = styles_arr == style_name
            sc = ax.scatter(
                projected[mask, 0], projected[mask, 1],
                c=y_rating[mask],
                cmap=CMAP,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.3,
                marker=style_to_marker[style_name],
                vmin=vmin,
                vmax=vmax,
            )
        sc_last = sc

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(panel_titles.get(instrument_name, instrument_name),
                     fontsize=10, fontweight="bold", pad=6)

        if SHOW_XLABEL[instrument_name]:
            ax.set_xlabel(f"{method_label} 1", fontsize=9)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if SHOW_YLABEL[instrument_name]:
            ax.set_ylabel(f"{method_label} 2", fontsize=9)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        ax.legend(
            handles=legend_handles, title="Style",
            fontsize=7, title_fontsize=8, loc="best",
        )

    if sc_last is not None:
        cbar = fig.colorbar(sc_last, cax=cbar_ax)
        cbar.set_label(rating, fontsize=9)

    if title_text is not None:
        fig.suptitle(title_text, fontsize=12, fontweight="bold", y=1.01)

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"3-row instrument figure saved to {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)


def plot_interactive_all_instruments_together(all_embeddings_dict, groove_ratings_dataframe,
                                              method="umap", ratings=DEFAULT_RATINGS,
                                              save_path=None, title_text=None):
    """
    Interactive HTML scatter plot (via Plotly) of all instruments in a single
    space, with hover tooltips showing the song name and all rating values.

    Parameters
    ----------
    all_embeddings_dict : dict[str, dict]
        instrument → per-song embedding dict.
    groove_ratings_dataframe : DataFrame
    method : str
    ratings : str or list[str]
        Rating column(s) to plot. A single string is accepted.
    save_path : str or None
    title_text : str or None
        Figure title; defaults to ``"<method> projection (All Instruments)"``.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from groovestim.ratings.compute_viz import _PLOTLY_SYMBOLS

    if isinstance(ratings, str):
        ratings = [ratings]

    projected, y_ratings, instruments, method_label, song_names = \
        _prepare_projection_all_instruments(
            all_embeddings_dict, groove_ratings_dataframe, ratings, method, annotate=True,
        )

    unique_instruments = sorted(set(instruments))
    instrument_to_symbol = {
        inst: _PLOTLY_SYMBOLS[i % len(_PLOTLY_SYMBOLS)] for i, inst in enumerate(unique_instruments)
    }

    # Hover text: song name + instrument + all requested ratings
    hover_texts = []
    for i, name in enumerate(song_names):
        parts = [f"<b>{name}</b>", f"instrument: {instruments[i]}"]
        for r in ratings:
            parts.append(f"{r}: {y_ratings[r][i]:.2f}")
        hover_texts.append("<br>".join(parts))

    fig = make_subplots(
        rows=1, cols=len(ratings),
        subplot_titles=list(ratings),
        horizontal_spacing=0.06,
    )

    for idx, rating_name in enumerate(ratings):
        for instrument_name in unique_instruments:
            mask = [inst == instrument_name for inst in instruments]
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
                        symbol=instrument_to_symbol[instrument_name],
                        colorbar=dict(
                            title=rating_name,
                            len=0.9,
                            x=1.0 if idx == len(ratings) - 1 else None,
                        ) if idx == len(ratings) - 1 else None,
                        line=dict(width=0.3, color="white"),
                    ),
                    text=[t for t, m in zip(hover_texts, mask) if m],
                    hoverinfo="text",
                    name=instrument_name,
                    legendgroup=instrument_name,
                    showlegend=(idx == 0),
                ),
                row=1, col=idx + 1,
            )
        fig.update_xaxes(title_text=f"{method_label} 1", row=1, col=idx + 1)
        fig.update_yaxes(title_text=f"{method_label} 2", row=1, col=idx + 1)

    fig.update_layout(
        height=400,
        width=350 * len(ratings),
        template="plotly_dark",
        title_text=title_text if title_text is not None else f"{method_label} projection (All Instruments)",
        margin=dict(t=60, b=40),
    )

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive figure with all instruments saved to {save_path}")

    return fig


def run_midi_drums_experiment(model, audio_folder_path, groove_ratings_dataframe,
                         checkpoint="default", seeds=None, viz_rating="groove"):
    """Run a midi_drums groove estimation for *model* over multiple seeds."""
    if seeds is None:
        seeds = SEEDS
    print(f"\n{'='*60}")
    print(f"[midi_drums] Computing embeddings for {model} (checkpoint={checkpoint})")
    print(f"{'='*60}")
    viz_tag = viz_rating.replace(" ", "_")

    song_audio_embeddings = load_embeddings(modality="midi_drums", model=model, checkpoint=checkpoint)

    compute_viz.save_viz(song_audio_embeddings, groove_ratings_dataframe, method="umap", ratings=[viz_rating], save_path=f"./viz/midi_drums/umap/{model}_{checkpoint}_{viz_tag}.png", title_text=f"UMAP | MIDI Drums")
    # compute_viz.save_viz(song_audio_embeddings, groove_ratings_dataframe, method="tsne", ratings=DEFAULT_RATINGS, save_path=f"./viz/midi_drums/tsne/{model}_{checkpoint}.png")
    compute_viz.save_interactive(song_audio_embeddings, groove_ratings_dataframe, method="umap", ratings=[viz_rating], save_path=f"./viz/midi_drums/umap/{model}_{checkpoint}_{viz_tag}.html", title_text=f"UMAP | MIDI Drums")
    # compute_viz.save_interactive(song_audio_embeddings, groove_ratings_dataframe, method="tsne", ratings=DEFAULT_RATINGS, save_path=f"./viz/midi_drums/interactive/{model}_{checkpoint}_tsne.html")
    # rsa.plot_distance_matrices_all_ratings(song_audio_embeddings, groove_ratings_dataframe, model_label=f"midi_drums_{model}_{checkpoint}")

    # R2 analysis estimation
    per_seed_results = []
    for seed in seeds:
        result = linear_probing.compute_ratings_scores(
            song_audio_embeddings, groove_ratings_dataframe,
            n_cross_val=N_CROSS_VAL, seed=seed,
        )
        per_seed_results.append(result)

    row = _results_to_row(
        per_seed_results, seeds,
        modality="midi_drums", model=model, checkpoint=checkpoint,
    )
    _print_model_results(row)
    return row

def run_source_separated_experiment(model, audio_folder_path, groove_ratings_dataframe, checkpoint="default", seeds=None, viz_rating="groove"):
    """Run a source_separated groove estimation for *model* over multiple seeds."""
    if seeds is None:
        seeds = SEEDS
    print(f"\n{'='*60}")
    print(f"[source_separated] Computing embeddings for {model} (checkpoint={checkpoint})")
    print(f"{'='*60}")
    viz_tag = viz_rating.replace(" ", "_")

    all_rows = []
    all_embeddings_dict = {}
    for chosen_instrument in INSTRUMENTS:
        song_audio_embeddings = load_embeddings(modality="source_separated", chosen_instrument=chosen_instrument, model=model, checkpoint=checkpoint)
        all_embeddings_dict[chosen_instrument] = song_audio_embeddings

        # compute_viz.save_viz(song_audio_embeddings, groove_ratings_dataframe, method="umap", ratings=[viz_rating], save_path=f"./viz/source_separated/{chosen_instrument}/umap/{model}_{checkpoint}_{viz_tag}.png", title_text=f"UMAP | {INSTRUMENTS_NAMES[chosen_instrument]}")
        compute_viz.save_viz(song_audio_embeddings, groove_ratings_dataframe, method="pca", ratings=[viz_rating], save_path=f"./viz/source_separated/for_nico/{chosen_instrument}/pca/{model}_{checkpoint}_{viz_tag}.png", title_text=f"PCA | {INSTRUMENTS_NAMES[chosen_instrument]}")
        # compute_viz.save_viz(song_audio_embeddings, groove_ratings_dataframe, method="tsne", ratings=DEFAULT_RATINGS, save_path=f"./viz/source_separated/{chosen_instrument}/tsne/{model}_{checkpoint}.png")
        # compute_viz.save_interactive(song_audio_embeddings, groove_ratings_dataframe, method="umap", ratings=[viz_rating], save_path=f"./viz/source_separated/{chosen_instrument}/umap/{model}_{checkpoint}_{viz_tag}.html", title_text=f"UMAP | {INSTRUMENTS_NAMES[chosen_instrument]}")
        # compute_viz.save_interactive(song_audio_embeddings, groove_ratings_dataframe, method="tsne", ratings=DEFAULT_RATINGS, save_path=f"./viz/source_separated/{chosen_instrument}/interactive/{model}_{checkpoint}_tsne.html")
        # rsa.plot_distance_matrices_all_ratings(song_audio_embeddings, groove_ratings_dataframe, model_label=f"source_separated_{chosen_instrument}_{model}_{checkpoint}")

        # R2 analysis estimation
        per_seed_results = []
        for seed in seeds:
            result = linear_probing.compute_ratings_scores(
                song_audio_embeddings, groove_ratings_dataframe,
                n_cross_val=N_CROSS_VAL, seed=seed,
            )
            per_seed_results.append(result)

        row = _results_to_row(
            per_seed_results, seeds,
            modality="source_separated", model=model, checkpoint=checkpoint, chosen_instrument=chosen_instrument
        )
        _print_model_results(row)
        all_rows.append(row)

        ground_truth = per_seed_results[0][viz_rating]['ground_truth']
        styles = io_utils._get_song_styles(song_audio_embeddings, groove_ratings_dataframe)

        per_seed_preds = [result[viz_rating]['prediction'] for result in per_seed_results]
        avg_preds = np.mean(per_seed_preds, axis=0)
        per_seed_r2 = [result[viz_rating]['R2'] for result in per_seed_results]
        avg_r2 = np.mean(per_seed_r2)
        linear_probing.save_scatter(ground_truth, avg_preds, rating_name=viz_rating, save_path=f"./viz/source_separated/for_nico/{chosen_instrument}/scatter/{model}_{checkpoint}_{viz_tag}_avg.png", styles=styles, R2_score=avg_r2)
        # linear_probing.save_scatter_interactive(ground_truth, avg_preds, groove_ratings_dataframe, rating_name=viz_rating, save_path=f"./viz/source_separated/{chosen_instrument}/scatter/{model}_{checkpoint}_{viz_tag}_avg.html", R2_score=avg_r2)

    # Plot all instruments combined using the same space mapping instruments to markers
    plot_all_instruments_together(
        all_embeddings_dict,
        groove_ratings_dataframe,
        method="umap",
        ratings=[viz_rating],
        save_path=f"./viz/source_separated/for_nico/all_instruments/umap/{model}_{checkpoint}_{viz_tag}.png"
    )
    plot_all_instruments_together(
        all_embeddings_dict,
        groove_ratings_dataframe,
        method="pca",
        ratings=[viz_rating],
        save_path=f"./viz/source_separated/for_nico/all_instruments/pca/{model}_{checkpoint}_{viz_tag}.png"
    )
    # plot_interactive_all_instruments_together(
    #     all_embeddings_dict,
    #     groove_ratings_dataframe,
    #     method="umap",
    #     ratings=DEFAULT_RATINGS,
    #     save_path=f"./viz/source_separated/all_instruments/umap/{model}_{checkpoint}_all_ratings.html"
    # )

    # 2×2 figure: bass/drums/other/vocals in one grid, per projection method
    for method in ["umap", "pca"]:
        plot_four_instruments_2x2(
            all_embeddings_dict,
            groove_ratings_dataframe,
            method=method,
            rating=viz_rating,
            save_path=f"./viz/source_separated/for_nico/2x2/{method}/{model}_{checkpoint}_{viz_tag}.png",
        )
        plot_five_instruments_3rows(
            all_embeddings_dict,
            groove_ratings_dataframe,
            method=method,
            rating=viz_rating,
            save_path=f"./viz/source_separated/for_nico/3rows/{method}/{model}_{checkpoint}_{viz_tag}.png",
        )

    return all_rows


# ── Dispatcher ───────────────────────────────────────────────────────────

def _run_modality(runner_fn, exp_config, audio_folder_path,
                  groove_ratings_dataframe, seeds=None, viz_rating="groove"):
    """Call *runner_fn* for every model/checkpoint and collect result rows."""
    if seeds is None:
        seeds = SEEDS
    all_rows = []
    for m, ckpts in exp_config["models"].items():
        for ckpt in ckpts:
            result = runner_fn(
                m, audio_folder_path, groove_ratings_dataframe,
                checkpoint=ckpt, seeds=seeds, viz_rating=viz_rating,
            )
            if isinstance(result, dict):
                all_rows.append(result)
            elif result:
                all_rows.extend(result)
    return all_rows


_RUNNERS = {
    "midi_drums": lambda exp_config, afp, grd, seeds, viz_rating: _run_modality(
        run_midi_drums_experiment, exp_config, afp, grd, seeds=seeds, viz_rating=viz_rating,
    ),
    "source_separated": lambda exp_config, afp, grd, seeds, viz_rating: _run_modality(
        run_source_separated_experiment, exp_config, afp, grd, seeds=seeds, viz_rating=viz_rating,
    ),
}


def run_experiments(experiments, viz_rating=VIZ_RATING):
    """
    Execute every experiment definition in *experiments*.

    Parameters
    ----------
    experiments : list[dict]
        Each dict must contain ``"modality"`` (one of ``"midi_drums"``,
        ``"source_separated"``) and ``"models"`` (a dict
        mapping model names to lists of checkpoint strings).
    """
    audio_folder_path, groove_ratings_dataframe = io_utils.load_common_data()

    all_rows = []
    for exp in experiments:
        modality = exp["modality"]
        runner = _RUNNERS.get(modality)
        if runner is None:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Choose from: {list(_RUNNERS.keys())}"
            )
        rows = runner(exp, audio_folder_path, groove_ratings_dataframe, seeds=SEEDS, viz_rating=viz_rating)
        all_rows.extend(rows)

    # ── Save aggregated results to CSV ──────────────────────────────
    if all_rows:
        results_df = pd.DataFrame(all_rows)
        os.makedirs(io_utils.DEFAULT_PLOT_DIR, exist_ok=True)
        csv_path = os.path.join(io_utils.DEFAULT_PLOT_DIR, "source_separated_experiment_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nAll results saved to {csv_path}")
        print_aggregated_results(results_df)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiments(EXPERIMENTS)
