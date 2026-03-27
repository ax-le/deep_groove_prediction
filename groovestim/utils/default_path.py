# Prpject folder
project_folder = '/Brain/private/a23marmo/projects/groove_study'

# Data folder
path_data_folder = f'{project_folder}/data' # '/home/nfarrugi/Documents/datasets/groove'

# Annotations
path_groove_ratings = f'{path_data_folder}/groove_raw_data/groove_ratings.xlsx' # '/home/nfarrugi/Documents/datasets/groove/list_songs_IDs_style_groove-rating_video-type.xlsx'
path_mir_features_xls = f'{path_data_folder}/groove_MIR_features.xlsx'
path_mir_features_csv = f'{path_data_folder}/groove_MIR_features.csv'

# Songs
path_audio_folder = f'{path_data_folder}/songs' # '/home/nfarrugi/Documents/datasets/groove/songs'

# Comments
path_groove_parquet = f'{path_data_folder}/groove.parquet'

# Cache
cache_dir_save_embeddings = f'{project_folder}/cache/embeddings'

# Model HuggingFace
cache_dir_huggingface = '/Brain/public/models' # '/home/nfarrugi/Documents/datasets/groove/checkpoints/'

