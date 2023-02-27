import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm

from config import data_config, feature_config
from utils import MovingAverage

DATA_PARAMS = data_config()
FEATURE_PARAMS = feature_config()

def process_file(filename):
    """
    Load a file and extract its features into a dataframe as a sequence of 
    overlapping spectrograms. Returned data frame will have the following content:
    magnitude: If feature type is set to a valid type, this will contain 2d feature data
    wave: If keep_wave is set to true, this will contain corresponding raw audio
    """
    # Get parameter data from config
    sr = DATA_PARAMS['sr']
    keep_wave = DATA_PARAMS['keep_wave']
    feature_type = FEATURE_PARAMS['type']
    n_fft = FEATURE_PARAMS['n_fft']
    win_length = FEATURE_PARAMS['win_length']
    hop_size = FEATURE_PARAMS['hop_size']
    n_bins = FEATURE_PARAMS['n_bins']
    chunk_size = FEATURE_PARAMS['chunk_size']
    overlap = FEATURE_PARAMS['overlap']
    padding = FEATURE_PARAMS['padding']

    # Compute derived parameters
    chunk_hop = chunk_size - overlap
    chunk_size_wav = chunk_size * hop_size
    overlap_wav = overlap * hop_size
    chunk_hop_wav = chunk_size_wav - overlap_wav

    df = dict()
    if feature_type != 'none': df['magnitude'] = []
    if keep_wave: df['wave'] = []
    try:
        # Load file
        audio, _ = librosa.load(filename, sr=sr)

        if feature_type != 'none':
            # Compute features
            if feature_type == 'stft':
                s = librosa.stft(audio, n_fft=n_fft-1, win_length=win_length, hop_length=hop_size)
            elif feature_type == 'cqt':
                s = librosa.cqt(audio, n_bins=n_bins, hop_length=hop_size)
            elif feature_type == 'mel':
                s = librosa.feature.melspectrogram(audio, n_fft=n_fft, win_length=win_length, hop_length=hop_size, n_mels=n_bins)
            else:
                raise Exception()

            # compute number of chunks
            num_chunks = max(1, -((overlap - s.shape[1]) // chunk_hop))

            # pad data (or alternatively truncate data)
            if padding:
                s = np.pad(s, [(0, 0), (0, num_chunks * chunk_hop + overlap - s.shape[1])])
            else:
                s = s[:,:(num_chunks - 1) * chunk_hop + overlap]

            # Create overlapping trunks along new dimension
            s = np.array([s[:,((chunk_hop) * i):((chunk_hop) * i) + chunk_size] for i in range(num_chunks)])

        if keep_wave:
            # Same steps as above
            num_chunks_wav = -((overlap_wav - audio.shape[0]) // chunk_hop_wav)
            if padding:
                audio = np.pad(audio, (0, num_chunks_wav * chunk_hop_wav + overlap_wav - audio.shape[0]))
            else:
                audio = audio[:(num_chunks_wav - 1) * chunk_hop_wav + overlap_wav]
            audio = np.array([audio[((chunk_hop_wav) * i):((chunk_hop_wav) * i) + chunk_size_wav] for i in range(num_chunks_wav)])

        # Apply log to spectrograms
        if feature_type == 'stft' or feature_type == 'cqt': mag = librosa.amplitude_to_db(np.absolute(s))
        if feature_type == 'mel': mag = librosa.power_to_db(s, ref=np.max)

        # Add to dataframe
        if keep_wave: df['wave'] += [audio]
        if feature_type != 'none': df['magnitude'] += [mag]
    except Exception as e:
        raise e
    return pd.DataFrame(df)


def process_files(dbdir=DATA_PARAMS['raw_data'], file_cap=None, min_sequence=DATA_PARAMS['min_sequence']):
    """
    iterates over audio files in a given directory and combines features into dataframe
    """
    file_types = DATA_PARAMS['file_types']

    # Find all files with the configured audio extensions
    files = []
    for root, dirnames, filenames in os.walk(dbdir):
        for filename in filenames:
            for extension in file_types:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
    
    # Iterate over files
    dfs = []
    mu = MovingAverage()
    with tqdm(files, desc='Computing Features', smoothing=0.1) as pbar:
        for i, f in enumerate(pbar):
            try:
                # Get features from file
                _df = process_file(f)

                # skip sequences shorter than the minimum sequence length
                if ('magnitude' in _df.iloc[0] and _df.iloc[0]['magnitude'].shape[0] < min_sequence or 
                    'wave' in _df.iloc[0] and _df.iloc[0]['wave'].shape[0] < min_sequence):
                    pbar.set_postfix(valid=mu.add(0))
                    continue
                else:
                    pbar.set_postfix(valid=mu.add(1))

                # Add file path as a column
                _df['file'] = [os.path.relpath(f, dbdir).replace("\\","/")]

                dfs += [_df]
            except Exception as e:
                raise e
            if file_cap is not None and i >= file_cap: break
    # Return the concatenation of all the files
    return pd.concat(dfs, ignore_index=True)


def get_directory(path, pos=DATA_PARAMS['category_level']):
    """
    get directory at specified position from a path
    """
    return os.path.normpath(path.replace("\\","/")).split(os.path.sep)[pos]


def save_hdf(df, dest=DATA_PARAMS['preprocessed_data'], min_cat_size=DATA_PARAMS['min_category_count'], single_file=DATA_PARAMS['single_file']):
    """
    save dataframe into an hdf5 file
    """
    category_level = DATA_PARAMS['category_level']
    label_key = DATA_PARAMS['hdf_label_key']

    # find the categories
    counts = df.apply(lambda x: get_directory(x['file']), axis=1).value_counts()
    categories = pd.DataFrame(counts[counts > min_cat_size].axes[0])[0]
    categories.to_hdf(dest + '.h5' if single_file else os.path.join(dest, label_key).replace("\\","/"), key=label_key)

    # Use a different key per label when saving to file
    for label in tqdm(categories, desc='Saving Dataset', smoothing=0.1):
        filtered_df = df[df.apply(lambda x: get_directory(x['file'], category_level), axis=1) == label]
        filtered_df.to_hdf(dest + '.h5' if single_file else os.path.join(dest, label).replace("\\","/"), key=label, mode='a')


if __name__ == "__main__":
    dbdir = DATA_PARAMS['raw_data']
    df = process_files(dbdir)

    dest = DATA_PARAMS['preprocessed_data']
    save_hdf(df, dest)