import os
from pathlib import Path
import argparse
import csv
import pickle

import numpy as np
from gglasso.problem import glasso_problem
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
import networkx as nx


# Path to the directory to save network data
MERTNET_DIR = os.getcwd() + "\\mertnet\\"

# Path to the MERT embeddings data matrix
EMBEDDINGS_PATH = MERTNET_DIR + 'embeddings.npy'

# Path to the MERT embeddings data matrix after PCA
EMBEDDINGS_PCA_PATH = MERTNET_DIR + 'embeddings_pca.npy'

# Path to a list of filenames in row-order of the embeddings matrix
EMBEDDINGS_FILENAMES = MERTNET_DIR + 'embeddings_filenames.pkl'

# Paths to the CSV files containing the overlaps and track features
OVERLAPS_DIR = os.getcwd() + "\\overlaps\\"
OVERLAPS_CSV = OVERLAPS_DIR + 'exact_overlaps.csv'
TRACK_FEATURES_CSV = OVERLAPS_DIR + 'track_features_with_genres.csv'

# Size of the hidden layer in the MERT network
N_FEATURES = 1024

# Selected hidden layer to use as the MERT embedding
layer = 25

# Paths to the inferred adjacency matrices
ADJACENCY_PCA_PATH = MERTNET_DIR + 'adjacency_pca.npy'
ADJACENCY_RAW_PATH = MERTNET_DIR + 'adjacency_raw_25.npy'

# Paths to save the constructed networks
NETWORK_PCA_PATH = MERTNET_DIR + 'mertnet_pca.pkl'
NETWORK_RAW_PATH = MERTNET_DIR + 'mertnet_raw.pkl'


def get_embeddings_subset(embeddings, layer):
    start = (layer - 1) * N_FEATURES
    end = start + N_FEATURES
    X = embeddings[:, start:end]
    return X


def run_glasso(data, N, lambda1_range=np.logspace(0, -3, 30), debug=False):
    if debug:
        lambda1_range = [0.1]
    
    print('Lambda range:', lambda1_range)

    print('Computing empirical covariance...')
    data = StandardScaler().fit_transform(data)
    S = empirical_covariance(data)
    print('Empirical covariance shape:', S.shape)
    
    sgl = glasso_problem(S, N, reg_params={'verbose': True}, 
                         latent=False, do_scaling=False)
    
    sgl.model_selection(modelselect_params={'lambda1': lambda1_range})

    sol = sgl.solution
    sol.calc_adjacency()

    return sol.precision_, sol.adjacency_


def create_dict_from_csv(csv_file, key_column):
    """Create a dictionary from a CSV file where the specified column is the key 
    and the remaining columns are the values.
    
    Args:
        csv_file (str): The path to the CSV file.
        key_column (str): The name of the column to use as the dictionary keys.
    
    Returns:
        dict: A dictionary where the keys are the values from the specified 
            column and the values are dictionaries of the remaining columns.
    """
    result_dict = {}
    
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.pop(key_column)
            result_dict[key] = row
    
    return result_dict


def get_node_attrs():
    node_attrs = {}

    with open(EMBEDDINGS_FILENAMES, 'rb') as f:
        filenames = pickle.load(f)
    
    overlaps_dict = create_dict_from_csv(OVERLAPS_CSV, 'Original Name')
    track_features_dict = create_dict_from_csv(TRACK_FEATURES_CSV, 'Song')

    for i, filename in enumerate(filenames):
        if filename not in overlaps_dict:
            print(f'Warning: {filename} not found in {OVERLAPS_CSV}')
            continue
        overlaps_dict[filename].pop('Original Row')
        node_attrs[i] = overlaps_dict[filename]
    
    for i in range(len(filenames)):
        trackname = node_attrs[i]['Song']
        if trackname not in track_features_dict:
            print(f'Warning: {trackname} not found in {TRACK_FEATURES_CSV}')
            continue
        node_attrs[i].update(track_features_dict[trackname])
    
    return node_attrs


def infer_network():
    mertnet_path = Path(MERTNET_DIR)
    mertnet_path.mkdir(parents=True, exist_ok=True)

    print('\nLoading MERT embeddings matrices...')
    embeddings = np.load(EMBEDDINGS_PATH)
    embeddings_pca = np.load(EMBEDDINGS_PCA_PATH)
    print('Embeddings shape:', embeddings.shape)
    print('Embeddings PCA shape:', embeddings_pca.shape)

    print('\nRunning graphical lasso with model selection on the PCA embeddings...')
    X_pca = embeddings_pca
    precision_pca, adjacency_pca = run_glasso(X_pca.T, N_FEATURES, debug=args.debug)
    print('Writing results to', mertnet_path)
    np.save(mertnet_path / f'precision_pca.npy', precision_pca)
    np.save(mertnet_path / f'adjacency_pca.npy', adjacency_pca)

    print('\nExtracting embeddings subset...')
    X = get_embeddings_subset(embeddings, layer)
    print('Embeddings subset shape:', X.shape)
    
    print('\nRunning graphical lasso with model selection on the raw embeddings...')
    precision_raw, adjacency_raw = run_glasso(X.T, N_FEATURES, debug=args.debug)

    print('Writing results to', mertnet_path)
    np.save(mertnet_path / f'precision_raw_{layer}.npy', precision_raw)
    np.save(mertnet_path / f'adjacency_raw_{layer}.npy', adjacency_raw)


def build_network():
    print('Getting node attributes...')    
    node_attrs = get_node_attrs()

    print('Building the networks...')
    A_pca = np.load(ADJACENCY_PCA_PATH)
    A_raw = np.load(ADJACENCY_RAW_PATH)
    G_pca = nx.from_numpy_array(A_pca)
    G_raw = nx.from_numpy_array(A_raw)
    nx.set_node_attributes(G_pca, node_attrs)
    nx.set_node_attributes(G_raw, node_attrs)

    print(f'Saving the networks to {MERTNET_DIR}...')
    with open(NETWORK_PCA_PATH, 'wb') as f:
        pickle.dump(G_pca, f)

    with open(NETWORK_RAW_PATH, 'wb') as f:
        pickle.dump(G_raw, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graphical lasso estimation.')
    parser.add_argument('--mode', choices=['infer', 'build'], required=True,
                        help='Specify whether to run graphical lasso or build the network from the result.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

    args = parser.parse_args()

    if args.debug:
        print('Debug mode enabled.')

    if args.mode == 'infer':
        infer_network()   
    elif args.mode == 'build':
        build_network()
