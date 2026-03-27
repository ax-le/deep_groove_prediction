import numpy as np
import pandas as pd

groove_keys_dataframe_col_correspondance = {
    'dance': 'Q1_dance-binned',
    'listen':'Q2_listen-binned',
    'beat':'Q3_beat-binned',
    'party': 'Q4_party-binned',
    'rhythm':'Q5_rhythm-binned', 
    'disturbing':'Q6_disturbing-binned',
    'groove':'groove-binned'
}

# make a function that generates a new column based on the value of an existing column (between 0 and 100), by binning the values in k bins
def bin_column(dataframe, column_name, new_column_name, k):
    # val = dataframe[column_name].values
    # bins = np.linspace(np.min(val), np.max(val), k+1)
    # dataframe[new_column_name] = pd.cut(dataframe[column_name], bins, labels=range(k), include_lowest=True)
    dataframe[new_column_name] = pd.qcut(dataframe[column_name], k, labels=False)
    return dataframe

def add_groove_binned_columns(dataframe_original, k):
    dataframe = dataframe_original.copy()
    dataframe = bin_column(dataframe, 'Q1_dance', 'Q1_dance-binned', k)
    dataframe = bin_column(dataframe, 'Q2_listen', 'Q2_listen-binned', k)
    dataframe = bin_column(dataframe, 'Q3_beat', 'Q3_beat-binned', k)
    dataframe = bin_column(dataframe, 'Q4_party', 'Q4_party-binned', k)
    dataframe = bin_column(dataframe, 'Q5_rhythm', 'Q5_rhythm-binned', k)
    dataframe = bin_column(dataframe, 'Q6_disturbing', 'Q6_disturbing-binned', k)
    dataframe = bin_column(dataframe, 'groove', 'groove-binned', k)
    return dataframe
