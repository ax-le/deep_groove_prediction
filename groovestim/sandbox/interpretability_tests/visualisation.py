import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import sklearn.manifold as manifold
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import cosine

# %% Study parameters
groove_var_subselection = ['groove_rating', 'dance', 'listen','party']#, 'Groove_0_100'] # 'Q3_beat', 'Q5_rhythm','Q6_disturbing', 
comment_selection_method = "song"
filter_kw = True
k_comments = 5
checkpoint_name = 'laion/larger_clap_general'

variance_threshold_selection = False

# %% Croos validation parameters
n_cross_val = 4
n_permutations_test = 100

# % Filtering function
def select_k_best_comments_for_this_song(song_id, audio_embeddings, comments_embeddings, song_ids_comments, comments_data_dataframe, comments_ids, k=5,filter_kw = False, comment_selection_method="song"):
    if comment_selection_method == "song":
        comment_row_indices = np.argwhere(song_ids_comments == song_id).reshape(-1)
    elif comment_selection_method == "all_comments":
        comment_row_indices = np.arange(comments_embeddings.shape[0])
    elif comment_selection_method == "random":
        filter_kw = False
        comment_row_indices = np.random.permutation(comments_embeddings.shape[0])[:k]
    elif comment_selection_method == "random_song":
        filter_kw = False
        song_indices = np.argwhere(song_ids_comments == song_id).reshape(-1)
        comment_row_indices = np.random.permutation(song_indices)[:k]
    else:
        raise ValueError(f"comment_selection_method not understood: {comment_selection_method}")
    
    # Just a check to make sure that the indices are the same as the comments_ids
    comments_ids_current_selection = comments_ids[comment_row_indices]
    assert (comments_ids_current_selection == comment_row_indices).all(), "Indices are the same as the comments_ids, because they were generated sequentially."

    if filter_kw:
        # keep only the rows with the following keys when they are 1
        filters = ['groove_bin', 'move_bin', 'flow_bin', 'power_bin', 'bonding_bin','sync_bin', 'event_bin']

        # generate a new column with the or between all the filters
        comments_data_dataframe['filter'] = comments_data_dataframe[filters].sum(axis=1)
        # keep only the indices from comment_row_indices that have a filter > 0
        filtered_rows = comments_data_dataframe.loc[comments_data_dataframe['filter']>0]
        filtered_comments_ids = filtered_rows['comment_id'].values

        # print(f"Before filtering: {comment_row_indices.shape}")
        filtered_comment_row_indices = np.intersect1d(comment_row_indices, filtered_comments_ids)
        # print(f"After filtering: {filtered_comment_row_indices.shape}")

        if filtered_comment_row_indices.size != 0:
            comment_row_indices = filtered_comment_row_indices
        else:
            print(f"No comments left for song {song_id}, using all comments instead.")

    if k is None or len(comment_row_indices) < k:
        # No need to select as it only remains less thank k comments.
        final_comment_indices = comment_row_indices
    else:
        this_song_audio_embedding = audio_embeddings[song_id].reshape(-1)
        distances = []

        for idx_comment in comment_row_indices:
            if comments_embeddings[idx_comment].shape[0] == 512:
                distances.append(cosine(comments_embeddings[idx_comment], this_song_audio_embedding))
        final_comment_indices = comment_row_indices[np.argsort(distances)[:k]]
    
    # Finding the rows of the selected comments
    rows_comments_dataframe = comments_data_dataframe.loc[comments_data_dataframe['comment_id'].isin(final_comment_indices)]
    # print(f"Song ID according to the code: {song_id}\nSelected comments: {rows_comments_dataframe}")
    best_comments = rows_comments_dataframe['comments'].values

    return final_comment_indices, best_comments

# %% Data loading
# Compute the list of audio files
audio_folder_path = paths.get_path_audio_folder()
song_list = [p.name for p in pathlib.Path(audio_folder_path).iterdir()]
song_ids = [pathlib.Path(s).stem[-12:-1] for s in song_list]

print(f"Loading ground truth and comments...")
# Loading metadata about the songs
groove_and_comments_dataframe = pd.read_parquet("./data/groove_clean.parquet")
# print(f"Some comments: {groove_and_comments_dataframe['comments'].values[:20]}")
# print(f"Songs of first comments: {groove_and_comments_dataframe['id'].values[:20]}")

print(f"Computing embeddings for {checkpoint_name}")
print(f"Loading text embeddings...")
file_comments_embeddings = np.load("./data/comments_embeddings.npz")
comments_embeddings = file_comments_embeddings['comments_embeddings']
comments_ids = file_comments_embeddings['comments_ids']
song_ids_comments = file_comments_embeddings['song_ids']
# print(song_ids[0])

print(f"Loading audio embeddings...")
audio_embeddings, song_list = clap_embed.get_audio_embeddings(audio_folder_path, checkpoint_name,paths.get_path_checkpoint()) # Compute and save the audio embeddings.

# %% Comment filtering
print("Filtering Youtube comments...")
average_embeddings = {}
text_embeddings = {}
for key in audio_embeddings.keys():
    comment_indexes, best_comments = select_k_best_comments_for_this_song(song_id=key, audio_embeddings=audio_embeddings, comments_embeddings=comments_embeddings, 
                                                                          song_ids_comments=song_ids_comments, comments_data_dataframe=groove_and_comments_dataframe,comments_ids=comments_ids,
                                                                          k=k_comments, filter_kw=filter_kw, comment_selection_method=comment_selection_method)
    text_embedding = np.mean(comments_embeddings[comment_indexes], axis=0)
    text_embeddings[key] = text_embedding.reshape(1, -1)
    this_song_audio_embedding = audio_embeddings[key].reshape(-1)
    test_mean = np.mean([this_song_audio_embedding, text_embedding], axis=0).reshape(1, -1)
    average_embeddings[key] = test_mean


manifold_method = "TSNE"
n_neighbors = 5
n_components = 2
metric = 'cosine'
min_dist = 0.1
n_epochs = 200
learning_rate = 0.1
random_state = 42
# %% Manifold learning
print("Computing manifold learning...")
if manifold_method == "UMAP":
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, min_dist=min_dist, n_epochs=n_epochs, learning_rate=learning_rate, random_state=random_state)
elif manifold_method == "TSNE":
    reducer = manifold.TSNE(n_components=n_components, metric=metric, random_state=random_state)
elif manifold_method == "Isomap":
    reducer = manifold.Isomap(n_components=n_components)
elif manifold_method == "PCA":
    reducer = PCA(n_components=n_components)
else:
    raise ValueError(f"Manifold method not understood: {manifold_method}")

# Compute the embeddings
X_audio_embeddings, ratings = groove_study.built_dataset(groove_and_comments_dataframe, audio_embeddings, col = 'party')
X_average_embeddings, ratings = groove_study.built_dataset(groove_and_comments_dataframe, average_embeddings, col = 'party')
X_textembeddings, ratings = groove_study.built_dataset(groove_and_comments_dataframe, text_embeddings, col = 'party')

X_low = reducer.fit_transform(np.vstack([X_average_embeddings, X_textembeddings]))


plt.subplot(1,2,1)
plt.scatter(X_low[:, 0], X_low[:, 1],c = np.concatenate((np.ones((X_average_embeddings.shape[0])), np.zeros((X_textembeddings.shape[0])))),alpha=0.6)

plt.subplot(1,2,2)

plt.scatter(X_low[:, 0], X_low[:, 1],c = np.concatenate((ratings,ratings)),alpha=0.6)
plt.colorbar()
plt.show()

