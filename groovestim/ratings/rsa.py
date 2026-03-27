"""
Representational Similarity Analysis (RSA) between song embeddings and groove ratings.

Compares pairwise distance structures: if songs that are close in embedding space
are also close in rating space, the Spearman correlation will be high.
"""

import pathlib

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from groovestim.ratings.reduce_dimensionality import rsa_select_k_best as reduce_embedding_dim_on_rsa_correlation
from groovestim.utils.io_utils import DEFAULT_RATINGS, _format_annotated_dataset, _format_annotated_dataset_sorted_by_rating

# %% Computing RSA
def compute_rsa(embeddings, ratings, embedding_metric='cosine', rating_metric='cityblock'):
    """
    Computes Representational Similarity Analysis (RSA) between embeddings and ratings.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n_samples, n_features)
    ratings : np.ndarray of shape (n_samples,) or (n_samples, 1)
    embedding_metric : str
        Distance metric for embeddings (e.g., 'cosine', 'euclidean').
    rating_metric : str
        Distance metric for ratings (e.g., 'euclidean', 'cityblock').

    Returns
    -------
    correlation : float
        Spearman rank correlation coefficient.
    p_value : float
        Two-sided p-value for the null hypothesis of no correlation.
    """
    # Ensure ratings are in a 2D column format for pdist
    if ratings.ndim == 1:
        ratings = ratings.reshape(-1, 1)

    # Compute pairwise distances
    dist_embeddings = pdist(embeddings, metric=embedding_metric)
    dist_ratings = pdist(ratings, metric=rating_metric)

    # Spearman rank correlation between the two distance vectors
    correlation, p_value = spearmanr(dist_embeddings, dist_ratings)

    return correlation, p_value




def compute_rsa_single_rating(
    song_embeddings, groove_ratings_dataframe, rating_name,
    embedding_metric='cosine', rating_metric='cityblock',
):
    """
    Compute RSA between song embeddings and a single groove rating.

    Parameters
    ----------
    song_embeddings : dict[str, np.ndarray]
        Mapping song_id → embedding array of shape (1, D).
    groove_ratings_dataframe : pd.DataFrame
        The groove study annotations (must contain columns 'id' and *rating_name*).
    rating_name : str
        Rating column to compare against (e.g. 'groove', 'Q1_dance').
    embedding_metric : str
        Distance metric for embeddings.
    rating_metric : str
        Distance metric for ratings.

    Returns
    -------
    correlation : float
        Spearman rank correlation coefficient.
    p_value : float
        Two-sided p-value.
    """
    X, y = _format_annotated_dataset(song_embeddings, groove_ratings_dataframe, rating_name)

    return compute_rsa(X, y, embedding_metric=embedding_metric, rating_metric=rating_metric)


def compute_rsa_all_ratings(
    song_embeddings, groove_ratings_dataframe,
    ratings=None,
    embedding_metric='cosine', rating_metric='euclidean',
):
    """
    Compute RSA between song embeddings and every groove rating.

    Parameters
    ----------
    song_embeddings : dict[str, np.ndarray]
        Mapping song_id → embedding array of shape (1, D).
    groove_ratings_dataframe : pd.DataFrame
        The groove study annotations.
    ratings : list[str] or None
        Rating columns to evaluate. Defaults to DEFAULT_RATINGS
        ('groove', 'Q1_dance', 'Q2_listen', 'Q4_party').
    embedding_metric : str
        Distance metric for embeddings.
    rating_metric : str
        Distance metric for ratings.

    Returns
    -------
    results : dict[str, dict]
        Mapping rating_name → {"correlation": float, "p_value": float}.
    """
    if ratings is None:
        ratings = DEFAULT_RATINGS

    results = {}
    for rating_name in ratings:
        correlation, p_value = compute_rsa_single_rating(
            song_embeddings, groove_ratings_dataframe, rating_name,
            embedding_metric=embedding_metric, rating_metric=rating_metric,
        )
        results[rating_name] = {
            "Spearman correlation": correlation,
            "p_value": p_value,
        }

    return results

# %% Plotting utils
def plot_distance_matrices(
    embeddings, ratings, rating_name,
    embedding_metric='cosine', rating_metric='cityblock',
    save_path=None, song_labels=None,
):
    """
    Plot the embedding and rating pairwise-distance matrices side by side.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n_samples, n_features)
    ratings : np.ndarray of shape (n_samples,) or (n_samples, 1)
    rating_name : str
        Used for the title and filename.
    embedding_metric / rating_metric : str
        Distance metrics forwarded to ``scipy.spatial.distance.pdist``.
    save_path : str or None
        If given, save the figure to this path.
    song_labels : list[str] or None
        If given, use as tick labels on both axes of both matrices.
    """
    if ratings.ndim == 1:
        ratings = ratings.reshape(-1, 1)

    dist_emb = squareform(pdist(embeddings, metric=embedding_metric))
    dist_rat = squareform(pdist(ratings, metric=rating_metric))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im1 = ax1.imshow(dist_emb, cmap='viridis', aspect='auto')
    ax1.set_title(f'Embedding distances ({embedding_metric})', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Song index')
    ax1.set_ylabel('Song index')
    fig.colorbar(im1, ax=ax1, shrink=0.85)

    im2 = ax2.imshow(dist_rat, cmap='viridis', aspect='auto')
    ax2.set_title(f'Rating distances – {rating_name} ({rating_metric})', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Song index')
    ax2.set_ylabel('Song index')
    fig.colorbar(im2, ax=ax2, shrink=0.85)

    if song_labels is not None:
        for ax in (ax1, ax2):
            ax.set_xticks(range(len(song_labels)))
            ax.set_yticks(range(len(song_labels)))
            ax.set_xticklabels(song_labels, rotation=90, fontsize=4)
            ax.set_yticklabels(song_labels, fontsize=4)

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Distance matrices saved to {save_path}')

    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)


def plot_distance_matrices_plotly(
    embeddings, ratings, rating_name, song_labels,
    embedding_metric='cosine', rating_metric='cityblock',
    save_path=None,
):
    """
    Interactive Plotly heatmaps of the pairwise-distance matrices.

    Hovering over a cell shows the two song names and the distance value.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n_samples, n_features)
    ratings : np.ndarray of shape (n_samples,) or (n_samples, 1)
    rating_name : str
        Used for titles.
    song_labels : list[str]
        Song names used as axis labels and hover text.
    embedding_metric / rating_metric : str
        Distance metrics forwarded to ``scipy.spatial.distance.pdist``.
    save_path : str or None
        Where to write the ``.html`` file.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if ratings.ndim == 1:
        ratings = ratings.reshape(-1, 1)

    dist_emb = squareform(pdist(embeddings, metric=embedding_metric))
    dist_rat = squareform(pdist(ratings, metric=rating_metric))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f'Embedding distances ({embedding_metric})',
            f'Rating distances – {rating_name} ({rating_metric})',
        ],
        horizontal_spacing=0.12,
    )

    for col_idx, (mat, title) in enumerate(
        [(dist_emb, 'Embedding'), (dist_rat, rating_name)], start=1,
    ):
        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=song_labels,
                y=song_labels,
                colorscale='Viridis',
                hovertemplate=(
                    '<b>%{x}</b> vs <b>%{y}</b><br>'
                    'Distance: %{z:.4f}<extra></extra>'
                ),
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        height=600,
        width=1400,
        template='plotly_dark',
        title_text=f'Self-similarity matrices – {rating_name}',
        margin=dict(t=80, b=40),
    )


    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f'Interactive distance matrices saved to {save_path}')

    return fig


def plot_distance_matrices_all_ratings(
    song_embeddings, groove_ratings_dataframe,
    ratings=None,
    embedding_metric='cosine', rating_metric='cityblock',
    save_dir='./viz/selfsimilaritymatrices',
    model_label='model',
    song_labels=True,
    interactive=True,
):
    """
    Plot distance matrices for every rating and save each to *save_dir*.

    Parameters
    ----------
    song_labels : bool
        If True, show song names as tick labels on the matplotlib plots.
    interactive : bool
        If True, also save interactive Plotly HTML files alongside the PNGs.
    """
    if ratings is None:
        ratings = DEFAULT_RATINGS

    for rating_name in ratings:
        X, y, names = _format_annotated_dataset_sorted_by_rating(
            song_embeddings, groove_ratings_dataframe,
            rating_name, sorting_rating='groove',
        )
        save_path = f"{save_dir}/{model_label}_{rating_name}.png"
        plot_distance_matrices(
            X, y, rating_name,
            embedding_metric=embedding_metric,
            rating_metric=rating_metric,
            save_path=save_path,
            song_labels=names if song_labels else None,
        )
        if interactive:
            html_path = f"{save_dir}/interactive/{model_label}_{rating_name}.html"
            plot_distance_matrices_plotly(
                X, y, rating_name, names,
                embedding_metric=embedding_metric,
                rating_metric=rating_metric,
                save_path=html_path,
            )


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
    results = compute_rsa_all_ratings(song_audio_embeddings, groove_ratings_dataframe, ratings=DEFAULT_RATINGS, embedding_metric='cosine', rating_metric='cityblock')
    print(results)
    plot_distance_matrices_all_ratings(song_audio_embeddings, groove_ratings_dataframe, model_label=f"{model}/{checkpoint}", song_labels=True, interactive=True)