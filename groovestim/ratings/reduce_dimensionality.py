"""
Dimensionality-reduction toolkit.

Provides a unified interface to standard and custom reduction methods.
Each public function follows the signature ``method(X, n_components=2, **kw) → np.ndarray``,
except for supervised methods (e.g. ``rsa_select_k_best``) which also accept *y*.

Use :func:`reduce` as a single entry-point dispatcher.
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import (
    PCA,
    FastICA,
    KernelPCA,
    TruncatedSVD,
)
from sklearn.feature_selection import SelectKBest
from sklearn.manifold import (
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
)
import umap


# ── Individual wrappers ─────────────────────────────────────────────────


def pca(X, n_components=2, **kwargs):
    """Principal Component Analysis."""
    return PCA(n_components=n_components, **kwargs).fit_transform(X)


def truncated_svd(X, n_components=2, random_state=0, **kwargs):
    """Truncated SVD (works on sparse matrices too)."""
    return TruncatedSVD(n_components=n_components, random_state=random_state, **kwargs).fit_transform(X)


def ica(X, n_components=2, random_state=0, **kwargs):
    """Independent Component Analysis (FastICA)."""
    return FastICA(n_components=n_components, random_state=random_state, **kwargs).fit_transform(X)


def kernel_pca(X, n_components=2, kernel="rbf", random_state=0, **kwargs):
    """Kernel PCA."""
    return KernelPCA(n_components=n_components, kernel=kernel, random_state=random_state, **kwargs).fit_transform(X)


def tsne(X, n_components=2, perplexity=5, metric="correlation", random_state=0, **kwargs):
    """t-distributed Stochastic Neighbour Embedding."""
    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        **kwargs,
    ).fit_transform(X)


def umap_reduce(X, n_components=2, n_neighbors=5, min_dist=0.3, metric="correlation", random_state=0, **kwargs):
    """UMAP projection."""
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **kwargs,
    ).fit_transform(X)


def isomap(X, n_components=2, n_neighbors=5, **kwargs):
    """Isomap (isometric feature mapping)."""
    return Isomap(
        n_components=n_components,
        n_neighbors=n_neighbors,
        **kwargs,
    ).fit_transform(X)


def lle(X, n_components=2, n_neighbors=5, random_state=0, **kwargs):
    """Locally Linear Embedding."""
    return LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
        **kwargs,
    ).fit_transform(X)


# ── RSA-based supervised feature selection ──────────────────────────────


def rsa_select_k_best(
    X, y,
    n_components=5,
    embedding_metric="cosine",
    rating_metric="cityblock",
    verbose=False,
):
    """
    Select the *n_components* embedding dimensions whose pairwise-distance
    structure best correlates (Spearman) with the rating pairwise distances.

    This is a supervised feature-selection method: it requires a target
    vector *y*.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    n_components : int
        Number of dimensions to keep.
    embedding_metric / rating_metric : str
        Distance metrics forwarded to ``scipy.spatial.distance.pdist``.
    verbose : bool
        Print a ranked summary when True.

    Returns
    -------
    X_reduced : np.ndarray of shape (n_samples, n_components)
        The input array restricted to the selected columns.
    """

    def _rsa_spearman_scorer(X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        dist_ratings = pdist(y, metric=rating_metric)

        n_dims = X.shape[1]
        scores = np.empty(n_dims)
        pvalues = np.empty(n_dims)

        for dim in range(n_dims):
            dist_emb = pdist(X[:, dim].reshape(-1, 1), metric=embedding_metric)
            corr, p_val = spearmanr(dist_emb, dist_ratings)
            scores[dim] = abs(corr)
            pvalues[dim] = p_val

        return scores, pvalues

    selector = SelectKBest(score_func=_rsa_spearman_scorer, k=n_components)
    selector.fit(X, y)
    X_reduced = selector.transform(X)

    if verbose:
        selected_indices = selector.get_support(indices=True)
        top_dims = [
            (int(idx), selector.scores_[idx], selector.pvalues_[idx])
            for idx in selected_indices
        ]
        top_dims.sort(key=lambda x: x[1], reverse=True)
        print(f"Top {n_components} embedding dimensions correlated with ratings:")
        for dim, corr, p_val in top_dims:
            print(f"  dim {dim:4d}:  |ρ| = {corr:.4f}  (p = {p_val:.2e})")

    return X_reduced


# ── Method registry & dispatcher ────────────────────────────────────────

_METHODS = {
    "pca":           pca,
    "truncated_svd": truncated_svd,
    "ica":           ica,
    "kernel_pca":    kernel_pca,
    "tsne":          tsne,
    "umap":          umap_reduce,
    "isomap":        isomap,
    "lle":           lle,
    # rsa_select_k_best is supervised and handled separately
}

METHOD_LABELS = {key: key.upper().replace("_", " ") for key in _METHODS}
METHOD_LABELS.update({
    "pca":           "PCA",
    "tsne":          "t-SNE",
    "umap":          "UMAP",
    "ica":           "ICA",
    "lle":           "LLE",
    "truncated_svd": "Truncated SVD",
    "kernel_pca":    "Kernel PCA",
    "isomap":        "Isomap",
})


def available_methods():
    """Return the list of registered unsupervised method names."""
    return list(_METHODS.keys())


def reduce_to_n_dims(X, method, n_components=2, gt_ratings=None, seed=0, **kwargs):
    """
    Dispatcher: apply *method* to reduce *X* to *n_components* dimensions.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    method : str
        One of :func:`available_methods`.
    n_components : int
        Target dimensionality.
    seed : int or None
        Random seed forwarded as ``random_state`` to stochastic methods
        for reproducibility.  Ignored by deterministic methods (pca, isomap).
    **kwargs
        Forwarded to the underlying method.

    Returns
    -------
    np.ndarray of shape (n_samples, n_components)
    """
    if method == "rsa_reduce":
        assert gt_ratings is not None, "Ground truth groove ratings must be provided for rsa selection"
        return rsa_select_k_best(X, y=gt_ratings, n_components=n_components, **kwargs)
    fn = _METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {available_methods()} "
            f"or call rsa_select_k_best() directly for supervised selection."
        )
    # Forward seed as random_state for methods that accept it
    if seed is not None and method not in ("pca", "isomap"):
        kwargs.setdefault("random_state", seed)
    return fn(X, n_components=n_components, **kwargs)
