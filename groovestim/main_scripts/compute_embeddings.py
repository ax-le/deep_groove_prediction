"""
Compute and inspect all embeddings used in groove experiments.

This script mirrors the structure of ``run_experiments.py`` but **only**
computes (or loads from cache) audio and text embeddings, without
running any rating estimation or plotting.  It prints verbose
diagnostics (shapes, dtypes, value ranges, timings) so you can
quickly verify that every embedding pipeline is working correctly.
"""

import itertools
import time

import numpy as np
import pandas as pd

import groovestim.utils.default_path as default_path
import groovestim.models.get_embeddings as get_embeddings
import groovestim.utils.comments_processing as comments_processing
import groovestim.utils.io_utils as io_utils


# ── Experiment configuration ────────────────────────────────────────────
EXPERIMENTS = [
    # {
    #     "modality": "text",
    #     "models": {
    #         "Qwen": ["default", "Qwen/Qwen3-Embedding-4B"],
    #         "EmbeddingGemma": ["default"],
    #         },
    #     "comment_filters": {
    #         "binary": [False],
    #         "closest": [False],
    #     },
    # },
    {
        "modality": "audio",
        "models": {
            "CLAP": ["laion/larger_clap_general", "laion/clap-htsat-fused", "laion/clap-htsat-unfused", "laion/larger_clap_music", "laion/larger_clap_music_and_speech"],
            }, #, "MERT-v1-95M","MERT-v1-330M"]},
    },
]

N_CLOSEST = 5


# ── Helpers ──────────────────────────────────────────────────────────────

def _print_embedding_summary(name, embeddings_dict):
    """Print shape / dtype / range for every song in *embeddings_dict*."""
    print(f"\n  ── {name} summary ({len(embeddings_dict)} songs) ──")
    for i, (song_id, emb) in enumerate(sorted(embeddings_dict.items())):
        emb = np.asarray(emb)
        print(
            f"    [{i:>3d}] song {song_id:>6s} | "
            f"shape {str(emb.shape):>20s} | "
            f"dtype {emb.dtype}"
        )

def _elapsed(t0):
    """Return a human-readable elapsed-time string since *t0*."""
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}min"


# ── Per-modality embedding computation ──────────────────────────────────

def compute_audio_embeddings(model, audio_folder_path, checkpoint="default"):
    """Compute audio embeddings for *model* and print diagnostics."""
    print(f"\n{'='*70}")
    print(f"[audio] Computing embeddings for model: {model}")
    print(f"  checkpoint = {checkpoint}")
    print(f"  audio_folder_path = {audio_folder_path}")
    print(f"{'='*70}")

    t0 = time.time()
    song_audio_embeddings = get_embeddings.audio_embeddings(
        model,
        audio_folder_path,
        checkpoint=checkpoint,
        space_mert="all",
    ) 
    print(f"  ✓ Audio embeddings computed in {_elapsed(t0)}")
    print(f"  Number of songs: {len(song_audio_embeddings)}")
    _print_embedding_summary(f"Audio ({model})", song_audio_embeddings)

    return song_audio_embeddings


def compute_text_embeddings(
    model, comment_filters,
    audio_folder_path,
    checkpoint="default",
):
    """Compute text embeddings for *model*, apply filters, print diagnostics."""
    print(f"\n{'='*70}")
    print(f"[text] Computing embeddings for model: {model}")
    print(f"  checkpoint = {checkpoint}")
    print(f"  comment_filters = {comment_filters}")
    print(f"{'='*70}")

    # ── Load text embeddings ─────────────────────────────────────────
    print("\n  Loading raw text embeddings...")
    t0 = time.time()
    song_text_embeddings = get_embeddings.text_embeddings(
        model, checkpoint=checkpoint,
    )
    print(f"  ✓ Text embeddings loaded in {_elapsed(t0)}")
    print(f"  Number of songs with comments: {len(song_text_embeddings)}")
    _print_embedding_summary(f"Raw text ({model})", song_text_embeddings)

    return song_text_embeddings


# ── Dispatcher ───────────────────────────────────────────────────────────

_RUNNERS = {
    "audio": lambda exp_config, audio_folder_path: [
        compute_audio_embeddings(m, audio_folder_path, checkpoint=ckpt)
        for m, ckpts in exp_config["models"].items()
        for ckpt in ckpts
    ],
    "text": lambda exp_config, audio_folder_path: [
        compute_text_embeddings(
            m, exp_config.get("comment_filters", {}), audio_folder_path,
            checkpoint=ckpt,
        )
        for m, ckpts in exp_config["models"].items()
        for ckpt in ckpts
    ],
    "audio_and_text": lambda exp_config, audio_folder_path: [
        compute_audio_and_text_embeddings(
            m, exp_config.get("comment_filters", {}), audio_folder_path,
            checkpoint=ckpt,
        )
        for m, ckpts in exp_config["models"].items()
        for ckpt in ckpts
    ],
}


def compute_all_embeddings(experiments):
    """
    Compute embeddings for every experiment definition in *experiments*.

    This is the embedding-only counterpart of
    :func:`run_experiments.run_experiments`.  It loads/computes all
    embeddings and prints detailed diagnostics, but does **not** run
    any rating estimation or plotting.

    Parameters
    ----------
    experiments : list[dict]
        Same format as ``run_experiments.EXPERIMENTS``.
    """
    print("=" * 70)
    print("  EMBEDDING COMPUTATION PIPELINE")
    print(f"  {len(experiments)} experiment group(s) to process")
    print("=" * 70)

    audio_folder_path, _ = io_utils.load_common_data()
    print(f"\n  Audio folder path: {audio_folder_path}")

    t_global = time.time()

    for i, exp in enumerate(experiments):
        modality = exp["modality"]
        print(f"\n\n{'#'*70}")
        print(f"# Experiment group {i + 1}/{len(experiments)}: modality = {modality}")
        print(f"# Models & checkpoints: {dict(exp['models'])}")
        if "comment_filters" in exp:
            print(f"# Comment filters: {exp['comment_filters']}")
        print(f"{'#'*70}")

        runner = _RUNNERS.get(modality)
        if runner is None:
            print(f"  ✗ Unknown modality '{modality}', skipping.")
            continue
        runner(exp, audio_folder_path)

    print(f"\n\n{'='*70}")
    print(f"  ALL DONE — total time: {_elapsed(t_global)}")
    print(f"{'='*70}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compute_all_embeddings(EXPERIMENTS)
