import os
from pathlib import Path
import argparse

import numpy as np
from gglasso.problem import glasso_problem
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance


# Path to the directory to save network data
MERTNET_DIR = os.getcwd() + "\\data\\mertnet"

# Path to the MERT embeddings data matrix
EMBEDDINGS_PATH = os.getcwd() + '\\data\\mertnet\\embeddings.npy'

# Path to the MERT embeddings data matrix after PCA
EMBEDDINGS_PCA_PATH = os.getcwd() + '\\data\\mertnet\\embeddings_pca.npy'

# Size of the hidden layer in the MERT network
N_FEATURES = 1024

# Selected hidden layer to use as the MERT embedding
layer = 25


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graphical lasso estimation.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

    args = parser.parse_args()

    if args.debug:
        print('Debug mode enabled.')

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
