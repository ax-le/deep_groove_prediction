import os
import pandas as pd
from tqdm import tqdm
import groovestim.utils.default_path as default_path
pd.set_option('display.max_columns', None)

drummer_files_path = "/Brain/private/a23marmo/projects/groove_study/data/data_from_olivier/data/data.csv"
drummer_dataframe = pd.read_csv(drummer_files_path)

groove_ratings_dataframe = pd.read_excel(default_path.path_groove_ratings)

# --- Check that song names are unique in groove_ratings_dataframe ---
groove_song_dupes = groove_ratings_dataframe["song"].duplicated(keep=False)
if groove_song_dupes.any():
    dup_songs = groove_ratings_dataframe.loc[groove_song_dupes, "song"].unique()
    raise ValueError(
        f"Duplicate song names found in groove_ratings_dataframe: {list(dup_songs)}. You should use the artist name as well to disambiguate."
    )
print("✓ Song names are unique in groove_ratings_dataframe. We can use the song name as key to merge.")

# --- Validation: ensure every song in groove_ratings_dataframe exists in drummer_dataframe ---
groove_songs = set(groove_ratings_dataframe["song"].unique())
drummer_songs = set(drummer_dataframe["track_title"].unique())

missing_from_drummer = groove_songs - drummer_songs
if missing_from_drummer:
    raise ValueError(
        f"The following songs from groove_ratings_dataframe are NOT found in "
        f"drummer_dataframe:\n{missing_from_drummer}"
    )
print(f"✓ All {len(groove_songs)} songs from groove_ratings_dataframe are present in drummer_dataframe.")

extra_in_drummer = drummer_songs - groove_songs
if extra_in_drummer:
    print(f"ℹ {len(extra_in_drummer)} songs in drummer_dataframe have no match in "
          f"groove_ratings_dataframe (this is expected):\n{extra_in_drummer}")

# --- Merge: add all columns from groove_ratings_dataframe to drummer_dataframe ---
# Inner join keeps only songs present in both dataframes
drummer_dataframe = drummer_dataframe.merge(
    groove_ratings_dataframe,
    left_on="track_title",
    right_on="song",
    how="inner",
)
# Rename "id" to "song_id" and drop the redundant "song" column from the merge
drummer_dataframe = drummer_dataframe.rename(columns={"id": "song_id"})
drummer_dataframe = drummer_dataframe.drop(columns=["song"])

print(f"\nResult — drummer_dataframe with groove ratings:\n{drummer_dataframe.head()}")
print(f"Total rows (present in both): {len(drummer_dataframe)}")
print(f"Columns: {list(drummer_dataframe.columns)}")

# --- Save to CSV ---
output_dir = os.path.dirname(drummer_files_path)
output_path = os.path.join(output_dir, "drummers_with_id_and_groove_ratings.csv")
drummer_dataframe.to_csv(output_path, index=False)
print(f"\n✓ Saved to {output_path}")
