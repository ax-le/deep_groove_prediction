"""
Shared helpers for the groove estimation scripts.

Centralises the data loading, rating estimation loop, and histogram plotting
that was previously duplicated across the three groove_estimation_from_*.py files.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pathlib
import torch
import torchaudio
import torchaudio.transforms as T

import groovestim.utils.default_path as default_path


# Default output directory for saved plots
DEFAULT_PLOT_DIR = './results'

# Supported audio file extensions
AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.flac', '.ogg', '.aac', '.wma'}

# Default ratings used across all estimation scripts
DEFAULT_RATINGS = ['groove', 'Q1_dance', 'Q2_listen', 'Q4_party']

# Default source separated instruments
SOURCE_SEPARATED_INSTRUMENTS = ["bass", "drums", "other", "vocals"]


# ── Song listing helpers ─────────────────────────────────────────────────

def extract_song_id(file_path):
    """
    Extract a song ID from the file path.

    Uses the stem (filename without extension) and takes the ``[-12:-1]``
    slice to match the ID format used in the MIR-features dataframe.
    """
    stem = pathlib.Path(file_path).stem
    return stem[-12:-1]

def get_song_list(audio_folder_path=default_path.path_audio_folder):
    """Return the list of song filenames in the audio folder."""
    return sorted([
        f.name for f in pathlib.Path(audio_folder_path).iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ])

def get_song_ids(audio_folder_path=default_path.path_audio_folder):
    """Return the list of song IDs in the audio folder."""
    # drums_weird = ['8iwBM_YB1sE', 'VlrQ-bOzpkQ', 'W5Opvi_UHLY', 'VbD_kBJc_gI']
    # all_ids = [extract_song_id(name) for name in get_song_list(audio_folder_path)]
    # return [id for id in all_ids if id not in drums_weird]
    return [extract_song_id(name) for name in get_song_list(audio_folder_path)]


# ── I/O utilities ──────────────────────────────────────────────────
# Cache for resamplers to avoid recreating them for each audio file
_resamplers = {}

def load_audio_file(audio_file_path, target_sr=48000):
    """
    Load an audio file, resample to *target_sr* if needed, and convert to mono.

    Parameters
    ----------
    audio_file_path : str or pathlib.Path
        Path to the audio file.
    target_sr : int
        Target sampling rate.  Defaults to 48000 (CLAP).  Use 24000 for MERT.

    Returns
    -------
    waveform : torch.Tensor
        1-D waveform tensor (mono).
    target_sr : int
        The sampling rate of the returned waveform.
    """
    waveform, base_sr = torchaudio.load(audio_file_path)

    # Resample if needed (cached resampler per source / target rate pair)
    if base_sr != target_sr:
        key = (base_sr, target_sr)
        if key not in _resamplers:
            _resamplers[key] = T.Resample(base_sr, target_sr)
        waveform = _resamplers[key](waveform)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=False)
    else:
        waveform = waveform.squeeze()

    return waveform, target_sr

def load_common_data():
    """
    Load the data shared by all groove estimation scripts:
    the audio folder path and the groove ratings dataframe.

    Returns
    -------
    audio_folder_path : str
    groove_ratings_dataframe : pd.DataFrame
    """
    audio_folder_path = default_path.path_audio_folder

    print("Loading groove ground truth dataframe...")
    groove_ratings_dataframe = pd.read_excel(default_path.path_groove_ratings)

    return audio_folder_path, groove_ratings_dataframe


def print_scores(results, title_prefix=""):
    """
    Print RMSE and R² scores for each rating.

    Parameters
    ----------
    results : dict[str, dict]
        Output of :func:`compute_ratings_scores`.
    title_prefix : str
        Prefix printed before each score line.
    """
    for rating_name, entry in results.items():
        print(
            f"{title_prefix}{rating_name}: "
            f"RMSE = {entry['RMSE']:.4f}, R² = {entry['R2']:.4f}"
        )


def plot_ratings(results, save_path=None):
    """
    Plot comparative histograms of ground truth vs. predictions,
    save the figure to *save_path*, and attempt to display it.

    Parameters
    ----------
    results : dict[str, dict]
        Output of :func:`compute_ratings_scores`.
    save_path : str or None
        File path to save the figure. If None, the figure is not saved.
    """
    ratings = list(results.keys())

    fig, axs = plt.subplots(
        1, len(ratings), constrained_layout=True, figsize=(10, 2)
    )

    for ax in axs.flat:
        ax.set(xlabel="Rating", ylabel="Number of songs")

    for idx, rating_name in enumerate(ratings):
        entry = results[rating_name]

        bins = (
            np.arange(-1, 1.1, 0.1)
            if rating_name == "groove"
            else range(0, 101, 5)
        )

        axs.flat[idx].hist(entry["ground_truth"], bins=bins, align="left", alpha=0.8)
        axs.flat[idx].hist(entry["prediction"], bins=bins, align="left", alpha=0.8)
        axs.flat[idx].title.set_text(rating_name)
        axs.flat[idx].set_ylim(0, 40)

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)


def _format_annotated_dataset(song_embeddings, groove_study_dataframe, rating = 'groove'):
    """
    Format the dataset for supervised learning between song embeddings (X) and groove ratings or other variables from the table (y)
    """
    X = []
    y = []
    # Sort keys to ensure deterministic ordering across runs
    for song_id in sorted(song_embeddings.keys()):
        groove_row = groove_study_dataframe.loc[groove_study_dataframe['id']==song_id]
        
        X.append(song_embeddings[song_id][0])
        y.append(float(groove_row[rating].values[0]))#groove_row[col].iloc[0]) #Other code: groove_row[col].values[0]
    X = np.stack(X)
    y = np.stack(y)
    return X, y

def _get_song_names(song_embeddings, groove_study_dataframe):
    """
    Return a list of song names aligned with the sorted song IDs used by
    the ``_format_annotated_dataset*`` functions.

    Falls back to the song ID if the ``'song'`` column is not present.
    """
    names = []
    for song_id in sorted(song_embeddings.keys()):
        row = groove_study_dataframe.loc[
            groove_study_dataframe["id"] == song_id
        ]
        if "song" in row.columns and len(row) > 0:
            names.append(str(row["song"].values[0]))
        else:
            names.append(str(song_id))
    return names


def _get_song_styles(song_embeddings, groove_study_dataframe):
    """
    Return a list of style labels aligned with the sorted song IDs used by
    the ``_format_annotated_dataset*`` functions.

    Falls back to ``"unknown"`` if the ``'style'`` column is missing or the
    row is empty.
    """
    styles = []
    for song_id in sorted(song_embeddings.keys()):
        row = groove_study_dataframe.loc[
            groove_study_dataframe["id"] == song_id
        ]
        if "style" in row.columns and len(row) > 0:
            styles.append(str(row["style_family"].values[0]))
        else:
            styles.append("unknown")
    return styles


def _format_annotated_dataset_sorted_by_rating(song_embeddings, groove_study_dataframe, rating='groove', sorting_rating='groove'):
    """
    Format the dataset sorted by rating value (ascending, least to most groovy).

    Identical to :func:`_format_annotated_dataset` but the rows are reordered
    so that the song with the lowest rating comes first and the song with the
    highest rating comes last.

    Parameters
    ----------
    song_embeddings : dict[str, np.ndarray]
        Mapping song_id → embedding array of shape (1, D).
    groove_study_dataframe : pd.DataFrame
        The groove study annotations (must contain columns 'id' and *rating*).
    rating : str
        Rating column to use as *y* (default ``'groove'``).
    sorting_rating : str
        Rating column to sort by (default ``'groove'``).

    Returns
    -------
    X : np.ndarray of shape (n_samples, D)
        Embeddings sorted by ascending sorting_rating.
    y : np.ndarray of shape (n_samples,)
        Rating values in ascending sorting_rating order.
    song_names : list[str]
        Song names in the same order.
    """
    X, y = _format_annotated_dataset(song_embeddings, groove_study_dataframe, rating)
    names = _get_song_names(song_embeddings, groove_study_dataframe)
    _, y_rating = _format_annotated_dataset(song_embeddings, groove_study_dataframe, sorting_rating)
    order = np.argsort(y_rating)
    sorted_names = [names[i] for i in order]
    return X[order], y[order], sorted_names


def _format_annotated_dataset_all_ratings(song_embeddings, groove_study_dataframe):
    """
    Format the dataset for supervised learning between song embeddings (X) and groove ratings or other variables from the table (y)
    """
    X = []
    y_ratings = {rating: [] for rating in DEFAULT_RATINGS}
    # Sort keys to ensure deterministic ordering across runs
    for song_id in sorted(song_embeddings.keys()):
        groove_row = groove_study_dataframe.loc[groove_study_dataframe['id']==song_id]
        
        X.append(song_embeddings[song_id][0])
        for rating in DEFAULT_RATINGS:
            y_ratings[rating].append(float(groove_row[rating].values[0]))
    X = np.stack(X)
    for rating in DEFAULT_RATINGS:
        y_ratings[rating] = np.stack(y_ratings[rating])
    return X, y_ratings