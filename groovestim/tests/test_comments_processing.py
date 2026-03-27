"""
Tests for groovestim.utils.comments_processing
"""

import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from groovestim.utils.comments_processing import (
    _BINARY_FILTERS,
    _get_binary_filtered_ids,
    _keep_closest,
    filter_comments,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_comments_df(n=10, seed=42):
    """
    Build a small comments DataFrame with *n* rows.

    Every row has a unique ``comment_id`` (0 … n-1), a ``comments`` text
    column, and the seven binary-flag columns.  By default the first half
    of the rows have *all* flags set to 0 (should be filtered out by
    binary filtering) and the second half have at least one flag set.
    """
    rng = np.random.RandomState(seed)
    data = {
        "comment_id": np.arange(n),
        "comments": [f"comment_{i}" for i in range(n)],
    }
    for col in _BINARY_FILTERS:
        flags = np.zeros(n, dtype=int)
        flags[n // 2 :] = rng.randint(0, 2, size=n - n // 2)
        data[col] = flags
    # Ensure at least one flag is set for the first second-half row
    data[_BINARY_FILTERS[0]][n // 2] = 1
    return pd.DataFrame(data)


def _make_embeddings(n=10, d=8, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float32)


def _make_audio_emb(d=8, seed=1):
    rng = np.random.RandomState(seed)
    return rng.randn(d).astype(np.float32)


# ── Tests for _get_binary_filtered_ids ───────────────────────────────────


class TestGetBinaryFilteredIds(unittest.TestCase):

    def test_returns_ids_with_at_least_one_flag(self):
        df = _make_comments_df()
        result = _get_binary_filtered_ids(df)
        for cid in result:
            row = df.loc[df["comment_id"] == cid].iloc[0]
            self.assertGreater(row[_BINARY_FILTERS].sum(), 0)

    def test_no_flags_set_returns_empty(self):
        df = _make_comments_df(n=6, seed=0)
        for col in _BINARY_FILTERS:
            df[col] = 0
        result = _get_binary_filtered_ids(df)
        self.assertEqual(len(result), 0)

    def test_all_flags_set_returns_all(self):
        df = _make_comments_df(n=5, seed=0)
        for col in _BINARY_FILTERS:
            df[col] = 1
        result = _get_binary_filtered_ids(df)
        np.testing.assert_array_equal(sorted(result), sorted(df["comment_id"].values))

    def test_single_flag_sufficient(self):
        """A row with exactly one flag set to 1 should be included."""
        df = _make_comments_df(n=3, seed=0)
        for col in _BINARY_FILTERS:
            df[col] = 0
        df.loc[1, "groove_bin"] = 1
        result = _get_binary_filtered_ids(df)
        self.assertIn(1, result)
        self.assertEqual(len(result), 1)


# ── Tests for _keep_closest ──────────────────────────────────────────────


class TestKeepClosest(unittest.TestCase):

    def test_returns_correct_number(self):
        emb = _make_embeddings()
        audio = _make_audio_emb()
        result = _keep_closest(audio, emb, np.arange(len(emb)), nb_closest=3)
        self.assertEqual(len(result), 3)

    def test_nb_closest_larger_than_pool(self):
        emb = _make_embeddings()
        audio = _make_audio_emb()
        result = _keep_closest(audio, emb, np.arange(len(emb)), nb_closest=50)
        self.assertEqual(len(result), len(emb))

    def test_ordering_is_by_cosine_distance(self):
        emb = _make_embeddings()
        audio = _make_audio_emb()
        local = np.arange(len(emb))
        result = _keep_closest(audio, emb, local, nb_closest=len(emb))
        distances = [cosine(e, audio.reshape(-1)) for e in emb]
        expected = np.argsort(distances)
        np.testing.assert_array_equal(result, expected)

    def test_respects_local_indices_subset(self):
        """Only the supplied local_indices should appear in the output."""
        emb = _make_embeddings()
        audio = _make_audio_emb()
        subset = np.array([1, 3, 7])
        result = _keep_closest(audio, emb[subset], subset, nb_closest=2)
        self.assertEqual(len(result), 2)
        for idx in result:
            self.assertIn(idx, subset)

    def test_works_with_2d_audio_embedding(self):
        """audio_embedding with shape (1, D) should be handled by reshape."""
        emb = _make_embeddings()
        audio_2d = np.ones((1, emb.shape[1]), dtype=np.float32)
        result = _keep_closest(audio_2d, emb, np.arange(len(emb)), nb_closest=2)
        self.assertEqual(len(result), 2)


# ── Tests for filter_comments (integration) ──────────────────────────────


class TestFilterComments(unittest.TestCase):

    def setUp(self):
        self.embeddings = _make_embeddings()
        self.comment_ids = np.arange(10)
        self.comments_df = _make_comments_df()
        self.audio_emb = _make_audio_emb()

    # --- no-filter baseline ---
    def test_no_filters_returns_all(self):
        sel_emb, sel_txt = filter_comments(
            self.embeddings, self.comment_ids, self.comments_df,
        )
        self.assertEqual(sel_emb.shape[0], len(self.comment_ids))
        self.assertEqual(len(sel_txt), len(self.comment_ids))

    # --- binary filtering ---
    def test_binary_filtering(self):
        sel_emb, sel_txt = filter_comments(
            self.embeddings, self.comment_ids, self.comments_df,
            binary_filtering=True,
        )
        expected_ids = _get_binary_filtered_ids(self.comments_df)
        self.assertEqual(sel_emb.shape[0], len(expected_ids))

    def test_binary_filtering_fallback_when_none_pass(self):
        """If no comment passes binary filtering, all should be kept."""
        df = _make_comments_df(n=10, seed=0)
        for col in _BINARY_FILTERS:
            df[col] = 0
        df["comment_id"] = np.arange(10)
        sel_emb, sel_txt = filter_comments(
            self.embeddings, self.comment_ids, df, binary_filtering=True,
        )
        self.assertEqual(sel_emb.shape[0], len(self.comment_ids))

    # --- closest filtering ---
    def test_closest_filtering(self):
        nb = 4
        sel_emb, sel_txt = filter_comments(
            self.embeddings, self.comment_ids, self.comments_df,
            audio_embedding=self.audio_emb, closest_filtering=True, nb_closest=nb,
        )
        self.assertEqual(sel_emb.shape[0], nb)
        self.assertEqual(len(sel_txt), nb)

    def test_closest_filtering_requires_audio_embedding(self):
        with self.assertRaises(ValueError):
            filter_comments(
                self.embeddings, self.comment_ids, self.comments_df,
                closest_filtering=True,
            )

    # --- random subset ---
    def test_random_subset(self):
        nb = 3
        sel_emb, sel_txt = filter_comments(
            self.embeddings, self.comment_ids, self.comments_df,
            random_subset=True, nb_random_comments=nb,
        )
        self.assertEqual(sel_emb.shape[0], nb)
        self.assertEqual(len(sel_txt), nb)

    def test_random_subset_different_seeds_give_different_results(self):
        results = []
        for seed in [10, 20]:
            np.random.seed(seed)
            sel_emb, _ = filter_comments(
                self.embeddings, self.comment_ids, self.comments_df,
                random_subset=True, nb_random_comments=5,
            )
            results.append(sel_emb)
        self.assertFalse(np.array_equal(results[0], results[1]))

    # --- combined filters ---
    def test_binary_then_closest(self):
        """Binary + closest: first prune by binary, then pick closest."""
        nb = 2
        sel_emb, sel_txt = filter_comments(
            self.embeddings, self.comment_ids, self.comments_df,
            audio_embedding=self.audio_emb,
            binary_filtering=True, closest_filtering=True, nb_closest=nb,
        )
        self.assertEqual(sel_emb.shape[0], nb)

    def test_random_then_binary(self):
        """Random subset applied before binary -> result can be smaller."""
        np.random.seed(0)
        sel_emb, sel_txt = filter_comments(
            self.embeddings, self.comment_ids, self.comments_df,
            random_subset=True, nb_random_comments=8, binary_filtering=True,
        )
        self.assertLessEqual(sel_emb.shape[0], 8)

    # --- output shapes ---
    def test_output_embedding_dimension_preserved(self):
        sel_emb, _ = filter_comments(
            self.embeddings, self.comment_ids, self.comments_df,
        )
        self.assertEqual(sel_emb.shape[1], self.embeddings.shape[1])

    # --- edge: single comment ---
    def test_single_comment(self):
        emb = np.random.randn(1, 4).astype(np.float32)
        cids = np.array([0])
        df = pd.DataFrame({
            "comment_id": [0],
            "comments": ["only comment"],
            **{col: [1] for col in _BINARY_FILTERS},
        })
        sel_emb, sel_txt = filter_comments(emb, cids, df)
        self.assertEqual(sel_emb.shape, (1, 4))
        self.assertEqual(sel_txt[0], "only comment")

    # --- edge: nb_random_comments > n_comments ---
    def test_random_subset_larger_than_pool(self):
        sel_emb, sel_txt = filter_comments(
            self.embeddings, self.comment_ids, self.comments_df,
            random_subset=True, nb_random_comments=100,
        )
        self.assertEqual(sel_emb.shape[0], len(self.comment_ids))


if __name__ == "__main__":
    unittest.main()
