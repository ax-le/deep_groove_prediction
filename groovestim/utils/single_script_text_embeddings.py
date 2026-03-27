
import pathlib
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


model_name = #TODO
checkpoint_name = #TODO

cache_dir_save_embeddings = '/Brain/private/a23marmo/projects/groove_study/cache/embeddings'
comments_parquet_path = "/Brain/private/a23marmo/projects/groove_study/data/groove_clean.parquet"
cache_dir_huggingface = f'/Brain/public/models/{model_name}'
model_path = f'/Brain/public/models/{checkpoint_name}'

# Device for inference (shared across all embedding modules)
device_general = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_name="default", cache_dir_huggingface=cache_dir_huggingface):
    # TODO
    return model, processor, checkpoint_name


def embed_this_text(text, model, processor):
    # model = model.to(device).eval()


    # TODO

    return embedding.detach().cpu().numpy()


def extract_song_id(file_path):
    """
    Extract a song ID from the file path.

    Uses the stem (filename without extension) and takes the ``[-12:-1]``
    slice to match the ID format used in the MIR-features dataframe.
    """
    stem = pathlib.Path(file_path).stem
    return stem[-12:-1]


def compute_or_load_text_embeddings(
    model_name,
    model,
    processor,
    embed_fn,
    checkpoint_name,
    comments_parquet_path=comments_parquet_path,
    batch_size=100,
    cache_dir=cache_dir_save_embeddings,
    verbose=True,
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

    # Load comments data and group by song
    comments_df = pd.read_parquet(comments_parquet_path)
    comments_by_song = {
        song_id: group[['comment_id', 'comments']]
        for song_id, group in comments_df.groupby('id')
    }

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


if __name__ == "__main__":

    model, processor, checkpoint = load_model(checkpoint_name)

    all_comments_embeddings, song_list = compute_or_load_text_embeddings(
        model_name,
        model,
        processor,
        embed_this_text,
        checkpoint,
        comments_parquet_path=comments_parquet_path,
        batch_size=100,
        cache_dir=cache_dir_save_embeddings,
        verbose=True,
    )
