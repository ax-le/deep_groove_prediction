"""
Handle comments: filtering, selection, etc.
"""

import numpy as np
from scipy.spatial.distance import cosine


def filter_comments(
    song_comment_data,
    comments_data_dataframe,
    audio_embedding=None,
    closest_filtering=False,
    nb_closest=5,
    binary_filtering=False,
    random_subset=False,
    nb_random_comments=5,
):
    """
    Filter the comments of a single song according to several criteria.

    Parameters
    ----------
    song_comment_data : dict
        Mapping ``{comment_id: embedding}`` where each embedding is a 1-D
        numpy array of shape ``(D,)``.
    comments_data_dataframe : pd.DataFrame
        The dataframe containing all comments and their binary filter columns.
    audio_embedding : np.ndarray or None
        Shape ``(D,)`` – this song's audio embedding, required when
        *closest_filtering* is True.
    closest_filtering : bool
        If True, keep the *nb_closest* comments that are closest (cosine)
        to *audio_embedding*.
    nb_closest : int
        Number of closest comments to keep when *closest_filtering* is True.
    binary_filtering : bool
        If True, keep only comments that have at least one binary flag set
        (groove_bin, move_bin, flow_bin, power_bin, bonding_bin, sync_bin,
        event_bin).
    random_subset : bool
        If True, randomly sample *nb_random_comments* from this song's
        comments (baseline mode).
    nb_random_comments : int
        Number of comments to sample when *random_subset* is True.

    Returns
    -------
    selected_data : dict
        Mapping ``{comment_id: embedding}`` for the selected comments.
    selected_comments : np.ndarray
        The text of the selected comments.
    """
    # Extract parallel arrays from the dict
    comment_ids = np.array(list(song_comment_data.keys()))
    comment_embeddings = np.array(list(song_comment_data.values()))

    # Start with all local indices for this song
    n_comments = len(comment_ids)
    local_indices = np.arange(n_comments)

    # Random baseline: subsample within this song's comments
    if random_subset:
        local_indices = np.random.permutation(local_indices)[:nb_random_comments]

    # Binary filtering: keep only comments with at least one binary flag
    if binary_filtering:
        filtered_cids = _get_binary_filtered_ids(comments_data_dataframe)
        # Intersect with this song's comment IDs at the selected indices
        mask = np.isin(comment_ids[local_indices], filtered_cids)
        if mask.any():
            local_indices = local_indices[mask]
        else:
            print(
                f"No comments left after binary filtering, "
                f"using all comments instead."
            )

    # Closest filtering: keep the N closest to the audio embedding
    if closest_filtering:
        if audio_embedding is None:
            raise ValueError(
                "audio_embedding must be provided when closest_filtering=True"
            )
        local_indices = _keep_closest(
            audio_embedding,
            comment_embeddings[local_indices],
            local_indices,
            nb_closest=nb_closest,
        )

    # Build the selected dict and retrieve comment texts
    selected_cids = comment_ids[local_indices]
    selected_data = {
        cid: song_comment_data[cid] for cid in selected_cids
    }

    rows = comments_data_dataframe.loc[
        comments_data_dataframe['comment_id'].isin(selected_cids)
    ]
    selected_comments = rows['comments'].values

    return selected_data, selected_comments


# ── Helpers ──────────────────────────────────────────────────────────────

_BINARY_FILTERS = [
    'groove_bin', 'move_bin', 'flow_bin',
    'power_bin', 'bonding_bin', 'sync_bin', 'event_bin',
]


def _get_binary_filtered_ids(comments_data_dataframe):
    """
    Return the comment IDs that have at least one binary flag set.
    """
    flag_sum = comments_data_dataframe[_BINARY_FILTERS].sum(axis=1)
    return comments_data_dataframe.loc[flag_sum > 0, 'comment_id'].values


def _keep_closest(audio_embedding, comment_embeddings, local_indices, nb_closest=5):
    """
    Return the *nb_closest* local indices whose embeddings are closest
    (cosine distance) to *audio_embedding*.
    """
    audio_flat = audio_embedding.reshape(-1)
    distances = [
        cosine(emb, audio_flat)
        for emb in comment_embeddings
    ]
    closest_order = np.argsort(distances)[:nb_closest]
    return local_indices[closest_order]
