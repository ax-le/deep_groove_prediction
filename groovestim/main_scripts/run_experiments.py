"""
Unified groove estimation experiment runner.

Replaces the three separate scripts (groove_estimation_from_audio.py,
groove_estimation_from_text.py, groove_estimation_from_text_and_audio.py)
with a single entry-point controlled by the EXPERIMENTS config below.
"""

import itertools
import os

import numpy as np
import pandas as pd

import groovestim.models.get_embeddings as get_embeddings
import groovestim.ratings.linear_probing as linear_probing
import groovestim.ratings.rsa as rsa
import groovestim.ratings.compute_viz as compute_viz

import groovestim.utils.default_path as default_path
import groovestim.utils.comments_processing as comments_processing
import groovestim.utils.io_utils as io_utils

# Global variable to control verbosity 
verbose = False
only_load_embeddings = True

MULTIMODAL_MODELS=["CLAP", "m2d", "OpenMuQ"]

from groovestim.utils.io_utils import DEFAULT_RATINGS

# ── Experiment configuration ────────────────────────────────────────────
EXPERIMENTS = [
    {
        "modality": "audio",
        "models": {
            "AudioMAE": ["hance-ai/audiomae"],
            "CLAP": ["laion/clap-htsat-unfused"],
            "m2d": ["m2d_CLAP"],
            "matpac": ["matpac_plus"],
            "MERT": ["MERT-v1-95M"],
            "musicfm": ["musicfm-25hz"],
            "OpenMuQ": ["MuQ-MuLan-large"],
            "MIR_features": ["default"],
        },
    },
    # {
    #     "modality": "text",
    #     "models": {
    #         "CLAP": ["laion/larger_clap_general"], #, "laion/clap-htsat-fused", "laion/clap-htsat-unfused", "laion/larger_clap_music"],
    #         "EmbeddingGemma": ["default"],
    #         "m2d": ["m2d_CLAP"],
    #         "OpenMuQ": ["MuQ-MuLan-large"],
    #         "Qwen": ["default"],
    #         "roberta": ["default"],
    #         },
    #     "comment_filters": {
    #         "binary": [False, True],
    #         "closest": [False, True],  # only multimodal models support closest filtering
    #     },
    # },
    # {
    #     "modality": "audio_and_text",
    #     "models": {
    #         "CLAP": ["laion/larger_clap_general"], #, "laion/clap-htsat-fused", "laion/clap-htsat-unfused", "laion/larger_clap_music"],
    #         "m2d": ["m2d_CLAP"],
    #         "OpenMuQ": ["MuQ-MuLan-large"]
    #         },
    #     "comment_filters": {
    #         "binary": [False, True],
    #         "closest": [False, True],
    #     },
    # },
]

N_CROSS_VAL = 4
N_CLOSEST = 5
SEEDS = [0, 1, 2, 3, 4]

# ── Shared helpers ──────────────────────────────────────────────────────


def _load_text_data(model, checkpoint, audio_folder_path, verbose=verbose, only_load_embeddings=only_load_embeddings):
    """
    Load text embeddings, validate song IDs against the audio folder,
    and load the comments dataframe.

    Returns
    -------
    song_emb : dict
        Text embeddings keyed by song ID.
    song_ids : list[str]
        Song IDs that have audio files (subset used for experiments).
    comments_df : pd.DataFrame
        The comments dataframe (needed for comment filtering).
    """
    print("Loading text embeddings...")
    song_emb = get_embeddings.text_embeddings(
        model, checkpoint=checkpoint, verbose=verbose, only_load_embeddings=only_load_embeddings,
    )
    text_song_ids = list(song_emb.keys())
    song_ids = io_utils.get_song_ids(audio_folder_path)
    print(f"Number of songs with audio: {len(song_ids)}")
    print(f"Number of songs with comments: {len(song_emb)}")

    assert all(sid in text_song_ids for sid in song_ids), \
        "All songs with audio must have comments."

    comments_df = pd.read_parquet("./data/groove_clean.parquet")
    return song_emb, song_ids, comments_df


def _filter_text_per_song(song_ids, song_emb, comments_df,
                          audio_embeddings, filter_closest, filter_binary,
                          fuse_audio=False):
    """
    For each song, filter comments and aggregate text embeddings.

    Parameters
    ----------
    audio_embeddings : dict or None
        Per-song audio embeddings.  Used for closest filtering regardless
        of *fuse_audio*.  When *fuse_audio* is True the audio embedding is
        also averaged with the text mean to produce the final embedding.
    fuse_audio : bool
        If True, average the audio and text embeddings for the final result.
    """
    result = {}
    for song_id in song_ids:
        audio_emb = audio_embeddings[song_id] if audio_embeddings else None
        selected_emb, _ = comments_processing.filter_comments(
            song_emb[song_id],
            comments_df,
            audio_embedding=audio_emb,
            closest_filtering=filter_closest,
            nb_closest=N_CLOSEST,
            binary_filtering=filter_binary,
        )
        emb_array = np.array(list(selected_emb.values()))
        text_mean = np.mean(emb_array, axis=0)

        if fuse_audio:
            audio_flat = audio_embeddings[song_id].reshape(-1)
            text_mean = np.mean([audio_flat, text_mean], axis=0)

        result[song_id] = text_mean.reshape(1, -1)
    return result


def _results_to_row(per_seed_results, seeds, modality, model, checkpoint,
                    binary_filter=None, closest_filter=None):
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
        "binary_filter": binary_filter,
        "closest_filter": closest_filter,
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
    if row.get("binary_filter") is not None:
        label_parts.append(f"bin={row['binary_filter']}")
    if row.get("closest_filter") is not None:
        label_parts.append(f"close={row['closest_filter']}")
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


# ── Per-modality runners ────────────────────────────────────────────────

def run_audio_experiment(model, audio_folder_path, groove_ratings_dataframe,
                         checkpoint="default", seeds=None):
    """Run a pure-audio groove estimation for *model* over multiple seeds."""
    if seeds is None:
        seeds = SEEDS
    print(f"\n{'='*60}")
    print(f"[audio] Computing embeddings for {model} (checkpoint={checkpoint})")
    print(f"{'='*60}")

    song_audio_embeddings = get_embeddings.audio_embeddings(
        model,
        audio_folder_path,
        checkpoint=checkpoint,
        space_mert="all",
        verbose=verbose,
        only_load_embeddings=only_load_embeddings,
    )

    compute_viz.save_viz(song_audio_embeddings, groove_ratings_dataframe, method="umap", ratings=DEFAULT_RATINGS, save_path=f"./viz/umap/{model}_{checkpoint}.png")
    compute_viz.save_viz(song_audio_embeddings, groove_ratings_dataframe, method="tsne", ratings=DEFAULT_RATINGS, save_path=f"./viz/tsne/{model}_{checkpoint}.png")
    compute_viz.save_comparison(song_audio_embeddings, groove_ratings_dataframe, ratings=DEFAULT_RATINGS, save_path=f"./viz/comparison/{model}_{checkpoint}.png")
    compute_viz.save_interactive(song_audio_embeddings, groove_ratings_dataframe, method="umap", ratings=DEFAULT_RATINGS, save_path=f"./viz/interactive/{model}_{checkpoint}_umap.html")
    compute_viz.save_interactive(song_audio_embeddings, groove_ratings_dataframe, method="tsne", ratings=DEFAULT_RATINGS, save_path=f"./viz/interactive/{model}_{checkpoint}_tsne.html")
    rsa.plot_distance_matrices_all_ratings(song_audio_embeddings, groove_ratings_dataframe, model_label=f"{model}_{checkpoint}")

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
        modality="audio", model=model, checkpoint=checkpoint,
    )
    _print_model_results(row)
    return row


def run_text_experiment(
    model, comment_filters,
    audio_folder_path, groove_ratings_dataframe,
    checkpoint="default",
    seeds=None,
):
    """Run text-only groove estimation for *model* with *comment_filters* over multiple seeds."""
    if seeds is None:
        seeds = SEEDS
    rows = []
    print(f"\n{'='*60}")
    print(f"[text] Model: {model} (checkpoint={checkpoint})")
    print(f"{'='*60}")

    song_emb, song_ids, comments_df = _load_text_data(
        model, checkpoint, audio_folder_path,
    )

    # ── Possibly load audio embeddings (needed for closest filtering) ─
    binary_options = comment_filters.get("binary", [False])
    closest_options = comment_filters.get("closest", [False])

    if model in MULTIMODAL_MODELS and True in closest_options:
        print("Loading audio embeddings (needed for closest filtering)...")
        audio_embeddings = get_embeddings.audio_embeddings(
            model, audio_folder_path, checkpoint=checkpoint,
            cache_dir_huggingface=default_path.cache_dir_huggingface,
            verbose=verbose,
            only_load_embeddings=only_load_embeddings,
        )
    else:
        audio_embeddings = None

    # ── Sweep over filter combinations ───────────────────────────────
    for filter_closest, filter_binary in itertools.product(
        closest_options, binary_options
    ):
        if filter_closest and model not in MULTIMODAL_MODELS:
            print(f"Closest filtering is only supported by {MULTIMODAL_MODELS}, skipping.")
            continue

        print(f"Binary filter: {filter_binary} / Closest filter: {filter_closest}")

        text_embeddings = _filter_text_per_song(
            song_ids, song_emb, comments_df,
            audio_embeddings if filter_closest else None,
            filter_closest, filter_binary,
            fuse_audio=False,
        )

        per_seed_results = []
        for seed in seeds:
            result = linear_probing.compute_ratings_scores(
                text_embeddings, groove_ratings_dataframe,
                n_cross_val=N_CROSS_VAL, seed=seed,
            )
            per_seed_results.append(result)

        row = _results_to_row(
            per_seed_results, seeds,
            modality="text", model=model, checkpoint=checkpoint,
            binary_filter=filter_binary, closest_filter=filter_closest,
        )
        _print_model_results(row)
        rows.append(row)

    return rows

def run_audio_and_text_experiment(
    model, comment_filters,
    audio_folder_path, groove_ratings_dataframe,
    checkpoint="default",
    seeds=None,
):
    """Run fused audio+text groove estimation for *model* over multiple seeds."""
    if seeds is None:
        seeds = SEEDS
    rows = []
    print(f"\n{'='*60}")
    print(f"[audio_and_text] Model: {model} (checkpoint={checkpoint})")
    print(f"{'='*60}")

    # ── Load audio embeddings ────────────────────────────────────────
    print("Loading audio embeddings...")
    audio_embeddings = get_embeddings.audio_embeddings(
        model, audio_folder_path, checkpoint=checkpoint,
        cache_dir_huggingface=default_path.cache_dir_huggingface,
        verbose=verbose, only_load_embeddings=only_load_embeddings,
    )

    # ── Load text data ───────────────────────────────────────────────
    song_emb, song_ids, comments_df = _load_text_data(
        model, checkpoint, audio_folder_path,
    )

    # ── Sweep over filter combinations ───────────────────────────────
    binary_options = comment_filters.get("binary", [False])
    closest_options = comment_filters.get("closest", [False])

    for filter_closest, filter_binary in itertools.product(
        closest_options, binary_options
    ):
        print(f"Binary filter: {filter_binary} / Closest filter: {filter_closest}")

        average_embeddings = _filter_text_per_song(
            song_ids, song_emb, comments_df,
            audio_embeddings,
            filter_closest, filter_binary,
            fuse_audio=True,
        )

        per_seed_results = []
        for seed in seeds:
            result = linear_probing.compute_ratings_scores(
                average_embeddings, groove_ratings_dataframe,
                n_cross_val=N_CROSS_VAL, seed=seed,
            )
            per_seed_results.append(result)

        row = _results_to_row(
            per_seed_results, seeds,
            modality="audio_and_text", model=model, checkpoint=checkpoint,
            binary_filter=filter_binary, closest_filter=filter_closest,
        )
        _print_model_results(row)
        rows.append(row)

    return rows


# ── Dispatcher ───────────────────────────────────────────────────────────

def _run_modality(runner_fn, exp_config, audio_folder_path,
                  groove_ratings_dataframe, seeds=None):
    """Call *runner_fn* for every model/checkpoint and collect result rows."""
    if seeds is None:
        seeds = SEEDS
    all_rows = []
    for m, ckpts in exp_config["models"].items():
        for ckpt in ckpts:
            result = runner_fn(
                m, audio_folder_path, groove_ratings_dataframe,
                checkpoint=ckpt, seeds=seeds,
            )
            if isinstance(result, dict):
                all_rows.append(result)
            elif result:
                all_rows.extend(result)
    return all_rows


_RUNNERS = {
    "audio": lambda exp_config, afp, grd, seeds: _run_modality(
        run_audio_experiment, exp_config, afp, grd, seeds=seeds,
    ),
    "text": lambda exp_config, afp, grd, seeds: _run_modality(
        lambda m, afp_, grd_, checkpoint, seeds: run_text_experiment(
            m, exp_config.get("comment_filters", {}),
            afp_, grd_, checkpoint=checkpoint, seeds=seeds,
        ),
        exp_config, afp, grd, seeds=seeds,
    ),
    "audio_and_text": lambda exp_config, afp, grd, seeds: _run_modality(
        lambda m, afp_, grd_, checkpoint, seeds: run_audio_and_text_experiment(
            m, exp_config.get("comment_filters", {}),
            afp_, grd_, checkpoint=checkpoint, seeds=seeds,
        ),
        exp_config, afp, grd, seeds=seeds,
    ),
}


def run_experiments(experiments):
    """
    Execute every experiment definition in *experiments*.

    Parameters
    ----------
    experiments : list[dict]
        Each dict must contain ``"modality"`` (one of ``"audio"``,
        ``"text"``, ``"audio_and_text"``) and ``"models"`` (a dict
        mapping model names to lists of checkpoint strings).
        Text-based modalities may also include a
        ``"comment_filters"`` dict with ``"binary"`` and ``"closest"``
        lists of booleans.
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
        rows = runner(exp, audio_folder_path, groove_ratings_dataframe, seeds=SEEDS)
        all_rows.extend(rows)

    # ── Save aggregated results to CSV ──────────────────────────────
    if all_rows:
        results_df = pd.DataFrame(all_rows)
        os.makedirs(io_utils.DEFAULT_PLOT_DIR, exist_ok=True)
        csv_path = os.path.join(io_utils.DEFAULT_PLOT_DIR, "experiment_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nAll results saved to {csv_path}")
        print_aggregated_results(results_df)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiments(EXPERIMENTS)
