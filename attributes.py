import os
import pickle
import csv
import argparse

import numpy as np
import networkx as nx


# Path to the directory to save network data
MERTNET_DIR = os.getcwd() + "\\mertnet\\"

# Path to a list of filenames in row-order of the embeddings matrix
EMBEDDINGS_FILENAMES = MERTNET_DIR + 'embeddings_filenames.pkl'

# Paths to the CSV files containing the overlaps and track features
OVERLAPS_DIR = os.getcwd() + "\\overlaps\\"
OVERLAPS_CSV = OVERLAPS_DIR + 'exact_overlaps.csv'
TRACK_FEATURES_CSV = OVERLAPS_DIR + 'track_features_with_genres.csv'


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


def build_network(adjacency_path, mode):
    print('Getting node attributes...')    
    node_attrs = get_node_attrs()

    print('Building the networks...')
    A = np.load(adjacency_path)
    G = nx.from_numpy_array(A)
    nx.set_node_attributes(G, node_attrs)

    print(f'Saving the networks to {MERTNET_DIR}...')
    save_path = MERTNET_DIR + f'mertnet_{mode}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(G, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process or load embeddings.')
    parser.add_argument('--mode', choices=['raw', 'pca', 'euclid'], required=True,
                        help='Choose which adjacency matrix to use.')

    args = parser.parse_args()

    build_network(MERTNET_DIR + f'adjacency_{args.mode}.npy', args.mode)
