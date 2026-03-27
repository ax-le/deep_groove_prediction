"""
High-level functions for loading and computing embeddings (audio and text).

This module is the single entry-point used by experiment scripts. It:
  - dispatches to the per-model modules in ``models/``,
  - provides shared utilities (audio loading, resampling, song-ID
    extraction, embedding caching) that the model modules also import.
"""

import pathlib

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

import groovestim.utils.default_path as default_path
import groovestim.utils.io_utils as io_utils


# ── Shared constants ─────────────────────────────────────────────────────

# Device for inference (shared across all embedding modules)
device_general = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── High-level API: audio embeddings ─────────────────────────────────────

def audio_embeddings(
    model_name,
    audio_folder_path,
    checkpoint="default",
    cache_dir_huggingface=default_path.cache_dir_huggingface,
    space_mert='all',
    verbose=True,
    only_load_embeddings=False,
):
    """
    Return audio embeddings for the given *model*.

    Parameters
    ----------
    model : str
        ``"MERT"``, ``"CLAP"``, or ``"MIR_features"``.
    audio_folder_path : str
        Path to the folder of audio files.
    checkpoint : str
        Checkpoint name (``"default"`` for each model's default).
    cache_dir_huggingface : str
        HuggingFace cache directory.
    space_mert : str or int
        MERT layer selection (``"all"`` or an integer index).
    """
    match model_name:
        case "MIR_features":
            from groovestim.models.compute_MIR_features import load_MIR_features as mir_get

            song_audio_embeddings, song_list = mir_get(audio_folder_path)
            return song_audio_embeddings

        case "MERT":
            from groovestim.models.compute_MERT_embeddings import load_model, embed_this_audio

            model_obj, processor, checkpoint = load_model(
                checkpoint_name=checkpoint,
                cache_dir_huggingface=cache_dir_huggingface,
            )

            song_audio_embeddings, song_list = compute_or_load_audio_embeddings(
                folder_path=audio_folder_path,
                model_name="MERT",
                model=model_obj,
                processor=processor,
                embed_fn=embed_this_audio,
                checkpoint_name=checkpoint,
                verbose=verbose,
                only_load_embeddings=only_load_embeddings,
            )

            def _select_mert_space(song_embeddings, space_mert):
                # Selecting one particular layer or all layers. Happens after embedding computation so that cached embeddings are not layer-specific.
                if space_mert == "all":
                    for sid in song_embeddings:
                        song_embeddings[sid] = song_embeddings[sid].reshape(1, -1)
                elif isinstance(space_mert, int):
                    n_layers = song_embeddings[next(iter(song_embeddings))].shape[0]
                    if space_mert >= n_layers:
                        raise ValueError(
                            f"The space index {space_mert} is larger than the "
                            f"number of MERT spaces ({n_layers})."
                        )
                    for sid in song_embeddings:
                        song_embeddings[sid] = song_embeddings[sid][space_mert, :].reshape(1, -1)
                else:
                    raise ValueError(
                        f"space_mert should be 'all' or an int, got: {space_mert}"
                    )
                return song_embeddings

            return _select_mert_space(song_audio_embeddings, space_mert)

        case "CLAP":
            from groovestim.models.compute_CLAP_embeddings import load_model, embed_this_audio

            model_obj, processor, checkpoint = load_model(
                checkpoint_name=checkpoint,
                cache_dir_huggingface=cache_dir_huggingface,
                modality="audio",
            )

        case "AudioMAE" | "MusicAesthetics" | "m2d" | "OpenMuQ" | "matpac" | "musicfm":
            if model_name == "MusicAesthetics":
                from groovestim.models.compute_MusicAesthetics_embeddings import load_model, embed_this_audio
            # if model_name == "AudioMAE":
                # from models.compute_AudioMAE_embeddings import load_model, embed_this_audio
            elif model_name in ["AudioMAE", "m2d", "OpenMuQ", "matpac", "musicfm"]:
                from groovestim.models.precomputed_embeddings_only import load_model, embed_this_audio

            model_obj, processor, checkpoint = load_model(
                checkpoint_name=checkpoint,
                cache_dir_huggingface=cache_dir_huggingface,
            )

        case _:
            raise ValueError(
                f"Model {model} not recognized. "
                f"Please use 'MERT', 'MIR_features', or 'CLAP'."
            )

    song_audio_embeddings, song_list = compute_or_load_audio_embeddings(
        folder_path=audio_folder_path,
        model_name=model_name,
        model=model_obj,
        processor=processor,
        embed_fn=embed_this_audio,
        checkpoint_name=checkpoint,
        verbose=verbose,
        only_load_embeddings=only_load_embeddings,
    )

    assert song_list == list(song_audio_embeddings.keys()), "Song IDs do not match between embeddings and song list loaded from directory"
    return song_audio_embeddings

# ── Generic audio-embedding caching loop ─────────────────────────────────

def compute_or_load_audio_embeddings(
    folder_path,
    model_name,
    model,
    processor,
    embed_fn,
    checkpoint_name,
    cache_dir=default_path.cache_dir_save_embeddings,
    verbose=True,
    only_load_embeddings=False,
):
    """
    Generic audio-embedding computation with on-disk caching.

    Loops over all audio files in *folder_path*, computes embeddings using
    the provided *embed_fn*, and caches them as ``.npz`` files.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Directory containing audio files.
    model_name : str
        Model name (e.g. ``"CLAP"``, ``"MERT"``), used for the cache sub-dir.
    model : torch.nn.Module
        The pre-loaded model.
    processor : object
        The pre-loaded processor / tokenizer.
    embed_fn : callable
        ``embed_fn(song_path, model, processor) -> embedding``.
    checkpoint_name : str
        Checkpoint identifier, used for the cache sub-dir.
    cache_dir : str or pathlib.Path
        Root cache directory for saved embeddings.

    Returns
    -------
    audio_embeddings : dict
        Mapping ``song_id → numpy embedding array``.
    song_ids_list : list
        List of song IDs extracted from audio filenames in *folder_path*.
    """
    folder = pathlib.Path(folder_path)

    # Filter to audio files only
    song_list = io_utils.get_song_list(folder_path)

    audio_embeddings = {}

    # Path of the folder where embeddings will be saved once computed
    dir_save_path = (
        pathlib.Path(cache_dir) / 'audio_embeddings' / model_name / checkpoint_name
    )

    # Loop on the songs
    for song_name in tqdm(song_list, desc=f"Computing {model_name} embeddings"):
        song_path = folder / song_name
        song_id = io_utils.extract_song_id(song_name)

        # Name of the file where the embedding will be saved
        embedding_file = dir_save_path / f'audio_embedding_{song_id}.npz'

        # Check if the embedding has already been computed
        if embedding_file.exists():
            this_embedding = np.load(embedding_file, allow_pickle=True)
            assert this_embedding['song_id'] == song_id
            this_audio_embedding = this_embedding['audio_embedding']
            if verbose:
                print(f"  ✓ Loaded audio embedding for song {song_id} from cache")
        else:
            if only_load_embeddings:
                raise NotImplementedError(f"Benchmarking mode, when only_load_embeddings=True, audio embeddings are not computed. It means that the path was not found, TODEBUG.")
            # Compute the embedding
            this_audio_embedding = embed_fn(str(song_path), model, processor)

            # Convert to numpy if it's a tensor
            if isinstance(this_audio_embedding, torch.Tensor):
                this_audio_embedding = this_audio_embedding.detach().cpu().numpy()

            dir_save_path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                embedding_file,
                song_id=song_id,
                audio_embedding=this_audio_embedding,
            )

            if verbose:
                print(f"  ✓ Computed audio embeddings for song {song_id}")
        
        audio_embeddings[song_id] = this_audio_embedding

    song_ids_list = [io_utils.extract_song_id(name) for name in song_list]
    return audio_embeddings, song_ids_list


# ── High-level API: text embeddings ──────────────────────────────────────

def text_embeddings(
    model_name,
    checkpoint="default",
    cache_dir_huggingface=default_path.cache_dir_huggingface,
    comments_parquet_path="./data/groove_clean.parquet",
    verbose=True,
    only_load_embeddings=False,
):
    """
    Return text (comment) embeddings for the given *model_name*.

    Uses :func:`get_text_embeddings` under the hood: if cached files
    exist the embeddings are loaded, otherwise they are computed and saved.

    Parameters
    ----------
    model_name : str
        ``"CLAP"`` or ``"roberta"``.
    checkpoint : str
        Checkpoint name (``"default"`` for each model's default).
    cache_dir_huggingface : str
        HuggingFace cache directory.
    comments_parquet_path : str
        Path to the cleaned comments parquet file.

    Returns
    -------
    song_comments_embeddings : dict
        ``{song_id: {comment_id: np.ndarray}}`` where each inner dict
        maps a comment ID to its embedding vector of shape ``(D,)``.
    """
    # Load comments data and group by song
    comments_df = pd.read_parquet(comments_parquet_path)
    comments_by_song = {
        song_id: group[['comment_id', 'comments']]
        for song_id, group in comments_df.groupby('id')
    }

    match model_name:

        case "CLAP":
            from groovestim.models.compute_CLAP_embeddings import load_model, embed_this_text

            model, processor, checkpoint = load_model(
                checkpoint_name=checkpoint,
                cache_dir_huggingface=cache_dir_huggingface,
                modality="text",
            )
            embed_fn = embed_this_text

        case "roberta" | "Qwen" | "EmbeddingGemma" | "m2d" | "OpenMuQ":
            if model_name == "roberta":
                from groovestim.models.compute_RoBERTa_embeddings import load_model, embed_this_text
            elif model_name == "Qwen":
                from groovestim.models.compute_Qwen_embeddings import load_model, embed_this_text
            elif model_name == "EmbeddingGemma":
                from groovestim.models.compute_EmbeddingGemma_embeddings import load_model, embed_this_text
            elif model_name in ["m2d", "OpenMuQ"]:
                from models.precomputed_embeddings_only import load_model, embed_this_text

            model, processor, checkpoint = load_model(
                checkpoint_name=checkpoint,
                cache_dir_huggingface=cache_dir_huggingface,
            )
            embed_fn = embed_this_text

        case _:
            raise ValueError(
                f"Text model '{model_name}' not recognized. "
                f"Please use 'CLAP', 'roberta', 'Qwen', 'EmbeddingGemma', 'm2d' or 'OpenMuQ'."
            )

    song_comments_embeddings, song_ids_list = compute_or_load_text_embeddings(
        model_name=model_name,
        comments_by_song=comments_by_song,
        model=model,
        processor=processor,
        embed_fn=embed_fn,
        checkpoint_name=checkpoint,
        verbose=verbose,
        only_load_embeddings=only_load_embeddings,
    )

    assert song_ids_list == list(song_comments_embeddings.keys()), "Song IDs do not match between embeddings and song list from comments csv"
    return song_comments_embeddings

# ── Generic text-embedding caching loop ──────────────────────────────────

def compute_or_load_text_embeddings(
    model_name,
    comments_by_song,
    model,
    processor,
    embed_fn,
    checkpoint_name,
    batch_size=100,
    cache_dir=default_path.cache_dir_save_embeddings,
    verbose=True,
    only_load_embeddings=False,
):
    """
    Per-song text-embedding computation with on-disk caching.

    For each song, all its comments are embedded (in batches of at most
    *batch_size*) and stored in a single ``.npz`` cache file.

    Parameters
    ----------
    model_name : str
        Model name (e.g. ``"CLAP"``, ``"roberta"``), used for the cache dir.
    comments_by_song : dict
        ``{song_id: DataFrame}`` where each DataFrame has columns
        ``'comment_id'`` and ``'comments'``.
    model : torch.nn.Module
        The pre-loaded model.
    processor : object
        The pre-loaded processor / tokenizer.
    embed_fn : callable
        ``embed_fn(texts_batch, model, processor) -> embedding tensor``.
    checkpoint_name : str
        Checkpoint identifier, used for the cache sub-dir.
    batch_size : int
        Maximum number of comments per forward pass (default 1000).
    cache_dir : str or pathlib.Path
        Root cache directory for saved embeddings.

    Returns
    -------
    all_comments_embeddings : dict
        ``{song_id: {comment_id: np.ndarray}}`` where each inner dict
        maps a comment ID to its embedding vector of shape ``(D,)``.
    song_ids_list : list
        Ordered list of song IDs matching the keys of
        *all_comments_embeddings*.
    """
    dir_save_path = (
        pathlib.Path(cache_dir) / 'text_embeddings' / model_name / checkpoint_name
    )

    all_comments_embeddings = {}

    song_ids_list = list(comments_by_song.keys())

    for song_id, song_df in tqdm(
        comments_by_song.items(), desc=f"Computing {model_name} text embeddings"
    ):
        cache_file = dir_save_path / f'text_embedding_{song_id}.npz'

        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            assert data['song_id'] == song_id
            text_embeddings_this_song = data['embeddings']
            comment_ids = data['comment_ids']
            if verbose:
                print(f"  ✓ Loaded text embeddings for song {song_id} from cache")
        else:
            if only_load_embeddings:
                raise NotImplementedError(f"Benchmarking mode, when only_load_embeddings=True, text embeddings are not computed. It means that the path was not found, TODEBUG.")
            texts = song_df['comments'].tolist()
            comment_ids = song_df['comment_id'].values

            # Embed in batches (songs can have >1000 comments)
            all_emb = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                emb = embed_fn(batch, model, processor)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                elif isinstance(emb, np.ndarray):
                    pass
                else:
                    raise ValueError(f"Embeddings must be torch.Tensor, got {type(emb)}")
                all_emb.append(emb)

            text_embeddings_this_song = np.vstack(all_emb)

            dir_save_path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_file,
                song_id=song_id,
                embeddings=text_embeddings_this_song,
                comment_ids=comment_ids,
            )
            if verbose:
                print(f"  ✓ Computed text embeddings for song {song_id}")

        dict_text_embeddings_this_song = {}
        for comment_id, embedding in zip(comment_ids, text_embeddings_this_song):
            dict_text_embeddings_this_song[comment_id] = embedding
        all_comments_embeddings[song_id] = dict_text_embeddings_this_song

    return all_comments_embeddings, song_ids_list

# ── Helper: flatten per-song dicts ───────────────────────────────────────

def flatten_song_embeddings(song_embeddings, song_comment_ids):
    """
    Flatten per-song dicts into parallel arrays, ordered by sorted song ID.

    Parameters
    ----------
    song_embeddings : dict[str, np.ndarray]
        ``{song_id: (n_comments, D)}`` embedding arrays.
    song_comment_ids : dict[str, np.ndarray]
        ``{song_id: (n_comments,)}`` comment-ID arrays.

    Returns
    -------
    embeddings : np.ndarray – shape ``(N, D)``
    song_ids : np.ndarray – shape ``(N,)``
    comment_ids : np.ndarray – shape ``(N,)``
    """
    all_emb, all_sid, all_cid = [], [], []
    for sid in sorted(song_embeddings):
        emb = song_embeddings[sid]
        cid = song_comment_ids[sid]
        all_emb.append(emb)
        all_sid.extend([sid] * len(emb))
        all_cid.append(cid)
    return (
        np.vstack(all_emb),
        np.array(all_sid),
        np.concatenate(all_cid),
    )

