import os
from pathlib import Path
import pickle
import re
import csv
import argparse

import requests
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import librosa
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from tqdm import tqdm

from wikimute.wikimute import WikiMuTe


# Device to use for model inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# User agent to use for Wikipedia requests
USER_AGENT = "RiceUDataBot/0.1.0"

# Maximum duration of the audio clips in seconds
MAX_DURATION_S = 30

# Path to the directory containing the MERT embeddings as .npy files
EMBEDDINGS_DIR = os.getcwd() + "\\data\\mert_embeds"

# Path to the CSV file containing the mapping from numpy file name to 
# the original WikiMuTe file name
IDX_TO_FILENAME_CSV = EMBEDDINGS_DIR + "\\progress.csv"

# Path to the CSV file containing the exact overlaps between the WikiMuTe 
# dataset and the Spotify Playlists dataset
OVERLAPS_CSV = os.getcwd() + "\\overlaps\\exact_overlaps.csv"

# Path to the directory to save network data
MERTNET_DIR = os.getcwd() + "\\data\\mertnet"


def sanitize_filename(filename):
    """Remove invalid characters for Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', '', filename)


def save_progress(progress_file, idx, file):
    """Save the index to file mapping to a CSV file."""
    with open(progress_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([idx, file])


def load_progress(progress_file):
    """Load the index to file mapping from the CSV file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            idx_to_file = {int(rows[0]): rows[1] for rows in reader}
        return idx_to_file
    return {}


def load_audio_file(file_path, resample_rate):
    """Load an audio file and resample it to the specified rate."""
    audio_data, sr = librosa.load(file_path, sr=resample_rate)

    if sr != resample_rate:
        raise ValueError(f'Unexpected sampling rate: {sr}')
    
    max_length = MAX_DURATION_S * sr
    if len(audio_data) > max_length:
        audio_data = audio_data[:max_length]
    
    return audio_data


def compute_mert_embedding(model, processor, audio_data, sr):
    """Compute the MERT embedding for an audio clip.

    For now, the mean across the time axis is used to reduce the dimension
    of the hidden states.
    
    Args:
        model (transformers.AutoModel): the MERT-v1-330M model
        processor (transformers.Wav2Vec2FeatureExtractor): the MERT-v1-330M
            feature extractor
        audio_data (np.ndarray): the audio data
        sr (int): the sampling rate of the audio data
    
    Returns:
        np.ndarray: the time-reduced MERT embedding, which has a shape of
            (25, 1024) corresponding to 25 hidden states and a feature
            dimension of 1024 
    
    """
    inputs = processor(audio_data,
                       sampling_rate=sr,
                       return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    time_reduced_hidden_states = all_layer_hidden_states.mean(-2)

    return time_reduced_hidden_states.cpu().numpy()


def process_wikimute_data(model, processor, df, temp_dir, output_dir, 
                          resample_rate, progress_file, debug=False):
    """Compute MERT embeddings for audio clips in the WikiMuTe dataset.

    MERT embeddings are saved as .npy files in a specified output directory.
    
    Args:
        model (transformers.AutoModel): the MERT-v1-330M model
        df (pandas.DataFrame): the WikiMuTe dataset
        temp_dir (str): the path a temporary directory to store audio files as
            they are processed
        output_dir (str): the path to the directory where the embeddings will
            saved as .npy files
        resample_rate (int): the sampling rate to resample the audio clips to
    
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    idx_to_file = load_progress(progress_file)
    processed_files = set(idx_to_file.values())
    
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    
    for audio_idx, (index, row) in enumerate(tqdm(df.iterrows(), 
                                                  total=df.shape[0], 
                                                  desc='Processing WikiMuTe data')):
        # Use the CSV file to keep track of previously processed files
        file = index
        if file in processed_files:
            print(f'Skipping {file}')
            continue
        url = row['audio_url']

        sanitized_file = sanitize_filename(file)
        temp_file_path = temp_path / sanitized_file
        
        # Download the audio file, or skip if the download fails
        downloaded = False
        try:
            req = session.get(url)
            req.raise_for_status()
            with open(temp_file_path, 'wb') as f:
                f.write(req.content)
            downloaded = True
        except requests.exceptions.HTTPError as err:
            print(f'Failed to download audio file: {file}')
            print(err.response.status_code)
            print(err.response.text)
            downloaded = False
        
        if not downloaded:
            if debug and audio_idx > 1:
                break
            continue

        loaded = False
        try:
            audio_data = load_audio_file(temp_file_path, resample_rate)
            loaded = True
        except ValueError as err:
            print(f'Failed to load audio file: {file}')
            print(err)
            loaded = False
        except Exception as err:
            print(f'Unknown error loading audio file: {file}')
            print(err)
            loaded = False
        
        if not loaded:
            if debug and audio_idx > 1:
                break
            continue
        
        # Compute the MERT embeddings for the audio clip
        embedding = compute_mert_embedding(model, 
                                           processor, 
                                           audio_data, 
                                           resample_rate)
        npy_file_path = output_path / (str(audio_idx) + '.npy')
        np.save(npy_file_path, embedding)

        os.remove(temp_file_path)
        save_progress(progress_file, audio_idx, file)

        if debug and audio_idx > 1:
            break
    
    os.rmdir(temp_path)


def get_overlap_set(overlaps_csv):
    """Get the set of overlapping tracks between the WikiMuTe dataset and the 
    Spotify Playlists dataset.
    
    Args:
        overlaps_csv (str): The path to the CSV file containing the exact overlaps.
    
    Returns:
        set: A set of tuples containing the artist and song name of the overlapping tracks.
    """
    overlap = set()
    with open(overlaps_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            overlap.add(row['Original Name'])
    return overlap


def get_idx_to_filename(idx_to_filename_csv):
    """Load the index to file mapping from the CSV file."""
    if os.path.exists(idx_to_filename_csv):
        with open(idx_to_filename_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            idx_to_file = {int(rows[0]): rows[1] for rows in reader}
        return idx_to_file
    return {}


def load_embeddings(dir, filtered=True):
    """Load all .npy files in the specified directory into a single matrix.
    
    Args:
        directory (str): The path to the directory containing the .npy files.
    
    Returns:
        np.ndarray: A single matrix containing all the embeddings.
    """
    overlap = get_overlap_set(OVERLAPS_CSV) if filtered else set()
    idx_to_file = get_idx_to_filename(IDX_TO_FILENAME_CSV)

    embeddings = []
    row_to_file = {}

    for idx, npy_file in enumerate(tqdm(os.listdir(dir),
                                    total=len(os.listdir(dir)),
                                    desc='Loading embeddings')):
        
        if not npy_file.endswith(".npy"):
            continue

        npy_file_idx = os.path.splitext(npy_file)[0]
        orig_file = idx_to_file[int(npy_file_idx)]

        if not filtered or orig_file in overlap:
            embedding = np.load(os.path.join(dir, npy_file))
            embedding = embedding.flatten()
            embeddings.append(embedding)
            row_to_file[idx] = orig_file

    X = np.vstack(embeddings)
    return X, row_to_file


def pca_reduce(matrix, n_components=.99):
    """Reduce the dimensionality of the embeddings using PCA and reduce the correlation between the vectors.
    
    Args:
        matrix (np.ndarray): The embeddings matrix.
        n_components (int or None): Number of components to keep. If None, all components are kept.
        plot_variance (bool): Whether to plot the explained variance ratio.
    
    Returns:
        np.ndarray: The transformed embeddings matrix.
    """
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    
    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(scaled_matrix)
    
    return pca_matrix


def preprocess():
    cwd = os.getcwd()
    PATH_TO_WIKIMUTE_DATA = cwd + '\\wikimute\\data\\all.csv'

    wikimute = WikiMuTe(PATH_TO_WIKIMUTE_DATA)
    dataset = wikimute.df

    temp_dir = cwd + '\\data\\temp'
    output_dir = cwd + '\\data\\mert_embeds'
    progress_file = output_dir + '\\progress.csv'

    model = AutoModel.from_pretrained('m-a-p/MERT-v1-330M', 
                                      trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained('m-a-p/MERT-v1-330M',
                                                         trust_remote_code=True)
    
    print('DEVICE:', DEVICE)
    model.to(DEVICE)

    resample_rate = processor.sampling_rate

    print('Processing the WikiMuTe data...')
    print('The resampling rate is:', resample_rate)

    process_wikimute_data(model, processor, dataset, temp_dir, output_dir, 
                          resample_rate, progress_file)

    print('Done processing WikiMuTe data.')


def load():
    mertnet_path = Path(MERTNET_DIR)
    mertnet_path.mkdir(parents=True, exist_ok=True)

    print("Loading MERT embeddings...")
    embeddings, filenames = load_embeddings(EMBEDDINGS_DIR)
    
    # The X matrix is the centered data matrix where each row is a component
    # of the MERT embedding and each column is a node in the inferred graph
    np.save(mertnet_path / 'embeddings.npy', embeddings)

    # Reduce the dimensionality of the embeddings using PCA
    print("Reducing dimensionality of embeddings by PCA...")
    embeddings_pca = pca_reduce(embeddings, n_components=.99)
    np.save(mertnet_path / 'embeddings_pca.npy', embeddings_pca)

    # Save the original filenames in the order of the rows of the data matrix
    with open(mertnet_path / 'embeddings_filenames.pkl', 'wb') as f:
        pickle.dump(filenames, f)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Process or load embeddings.')
    parser.add_argument('--mode', choices=['preprocess', 'load'], required=True,
                        help='Specify whether to run the preprocess or load_embeddings function.')
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        preprocess()
    elif args.mode == 'load':
        load()
