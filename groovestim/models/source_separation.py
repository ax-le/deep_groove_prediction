import demucs.separate
import groovestim.utils.io_utils as io_utils
import groovestim.utils.default_path as default_path

audio_folder_path=default_path.path_audio_folder

def run_source_separation(song_path, stored_data_folder, model = 'htdemucs_ft', verbose = True):
    demucs.separate.main(["--name", model, "-o", f"{stored_data_folder}/", song_path])

    if verbose:
        print("Source separation done.")

def run_source_separation_two_stems(song_path, stored_data_folder, instrument_name, model = 'htdemucs_ft', verbose = True):
    demucs.separate.main(["--two-stems", instrument_name, "--name", model, "-o", f"{stored_data_folder}/two_stems", song_path])

    if verbose:
        print("Source separation done.")


if __name__ == "__main__":
    for song in io_utils.get_song_list(audio_folder_path):
        for instrument_name in ["vocals", "drums", "bass", "other"]:
            run_source_separation_two_stems(f"{audio_folder_path}/{song}", "/Brain/private/a23marmo/projects/groove_study/data/two_stems_source_separated", instrument_name)