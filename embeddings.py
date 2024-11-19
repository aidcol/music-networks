import os
from pathlib import Path
import re
import csv

import requests
import numpy as np
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


def process_wikimute_data(model, df, temp_dir, output_dir, resample_rate,
                          progress_file, debug=False):
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


if __name__ == '__main__':    
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

    process_wikimute_data(model, dataset, temp_dir, output_dir, resample_rate, 
                          progress_file)

    print('Done processing WikiMuTe data.')
