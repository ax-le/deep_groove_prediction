from scipy.spatial.distance import cosine
import numpy as np
from sklearn.dummy import DummyClassifier
import bin_ratings as br



def zero_shot_groove_bin_this_song(song_embedding, groove_text_embeddings):
    answers = {}
    for groove_key in groove_text_embeddings.keys():
        distances = []
        for a_text_embed in groove_text_embeddings[groove_key]:
            distances.append(cosine(song_embedding,a_text_embed))        
        answers[groove_key] = np.argmin(distances)
    return answers

def zero_shot_groove_bin_evaluation(song_embeddings, sentences_embeddings, song_ids, groove_study_dataframe, groove_keys):
    estimated_groove_ratings = {}
    groove_error = -np.inf * np.ones((len(song_ids), len(groove_keys))) # Je stocke dans un tableau numpy parce que je (Axel) suis plus habitué à faire des stats dessus.
    groove_accuracy = -np.inf * np.ones((len(song_ids), len(groove_keys))) # Je stocke dans un tableau numpy parce que je (Axel) suis plus habitué à faire des stats dessus.

    # Select the subset of columns correpsonding to the keys to evaluate
    dataframe_col_subset = [br.groove_keys_dataframe_col_correspondance[cur_key] for cur_key in groove_keys]

    for song_iterator, song_id in enumerate(song_ids):
        ground_truth_groove_rating = groove_study_dataframe.loc[groove_study_dataframe['id']==song_id]
        estimation_of_groove_rating = zero_shot_groove_bin_this_song(song_embeddings[song_id][0], sentences_embeddings)

        estimated_groove_ratings[song_id] = estimation_of_groove_rating

        for i in range(len(groove_keys)):
            groove_error[song_iterator][i] = np.linalg.norm(estimation_of_groove_rating[groove_keys[i]] - ground_truth_groove_rating[dataframe_col_subset[i]].values[0])
            groove_accuracy[song_iterator][i] = 1 if estimation_of_groove_rating[groove_keys[i]] == ground_truth_groove_rating[dataframe_col_subset[i]].values[0] else 0

    return estimated_groove_ratings, groove_error, groove_accuracy

def baseline_dummy_classifier(groove_study_dataframe, groove_keys):
    dummy_clf = DummyClassifier(strategy="prior")
    accuracies = -np.inf*np.ones(len(groove_keys))

    for i in range(len(groove_keys)):
        groove_key_df = br.groove_keys_dataframe_col_correspondance[groove_keys[i]]
        labels = groove_study_dataframe[groove_key_df].values
        dummy_data = np.zeros(len(labels))
        dummy_clf.fit(dummy_data, labels)
        dummy_clf.predict(dummy_data)
        dummy_accuracy = dummy_clf.score(dummy_data, labels)
        accuracies[i] = dummy_accuracy
    return accuracies