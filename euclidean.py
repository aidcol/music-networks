import os

from tqdm import tqdm
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


MERTNET_DIR = os.getcwd() + '\\mertnet\\'
EMBEDDINGS_PATH = MERTNET_DIR + 'embeddings.npy'
DISTANCE_HISTOGRAM_PATH = MERTNET_DIR + 'distance_histogram.png'
CONNECTIVITY_V_THRESHOLD_PATH = MERTNET_DIR + 'connectivity_v_threshold.png'
ADJACENCY_PATH = MERTNET_DIR + 'adjacency_euclid.npy'
LAYER = 25


def get_embeddings_subset(embeddings, layer):
    n_features = 1024
    start = (layer - 1) * n_features
    end = start + n_features
    X = embeddings[:, start:end]
    return X


def threshold_distance_matrix(distance_matrix, threshold):
    """
    Apply a threshold to a distance matrix to drop large distances.
    
    Args:
        distance_matrix (ndarray): Pairwise distance matrix of shape (n_samples, n_samples).
        threshold (float): Threshold value; distances greater than this will be set to infinity.
    
    Returns:
        ndarray: Thresholded distance matrix of shape (n_samples, n_samples).
    """
    thresholded_matrix = distance_matrix.copy()
    thresholded_matrix[thresholded_matrix > threshold] = -1
    return thresholded_matrix


def compute_similarity_weights(distance_matrix, sigma):
    """
    Compute similarity weights using a Gaussian kernel based on a distance matrix.
    
    Args:
        distance_matrix (ndarray): Thresholded distance matrix of shape (n_samples, n_samples).
        sigma (float): Gaussian kernel width parameter.
    
    Returns:
        ndarray: Similarity matrix of shape (n_samples, n_samples).
    """
    similarity_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))
    similarity_matrix[distance_matrix < 0] = 0
    return similarity_matrix


def get_connectivity_versus_threshold(distances, n_points=100):
    """Compute relative size of the largest connected component as a function 
    of the threshold.
    
    Args:
        distances (np.ndarray): Square distance matrix
        initial_threshold (float): Initial threshold value
        step (float): Step size for threshold
    
    Returns:
        float: Selected threshold value
    """
    num_samples = len(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    threshold_range = np.linspace(min_distance, max_distance, n_points)

    size_threshold_pairs = []
    sigma = np.median(distances)
    print('Sigma for Gaussian kernel chosen as the median observed distance:', 
          sigma)

    for threshold in tqdm(threshold_range):
        thresholded_matrix = threshold_distance_matrix(distances, threshold)
        adjacency_matrix = compute_similarity_weights(thresholded_matrix, sigma)
        G = nx.from_numpy_array(adjacency_matrix)
        largest_component = max(nx.connected_components(G), key=len)
        relative_size = len(largest_component) / num_samples
        size_threshold_pairs.append((relative_size, threshold))

    return size_threshold_pairs


def select_threshold(size_threshold_pairs, min_size=0.99):
    """Select the threshold value corresponding to a minimum relative size of the largest connected component.
    
    Args:
        size_threshold_pairs (list): List of (relative size, threshold) pairs
        min_size (float): Minimum relative size
    
    Returns:
        float: Selected threshold value
    """
    for size, threshold in size_threshold_pairs:
        if size >= min_size:
            return threshold, size
    return None


def construct_graph(X, threshold):
    """
    Construct a graph from a data matrix X using distance thresholding and Gaussian kernel similarity weights.
    
    Args:
        X (ndarray): Data matrix of shape (n_samples, n_features).
        threshold (float): Distance threshold for dropping large distances.
        sigma (float): Gaussian kernel width parameter.
    
    Returns:
        ndarray: Graph adjacency matrix of shape (n_samples, n_samples).
    """
    distance_matrix = squareform(pdist(X, 'euclidean'))
    sigma = np.median(distance_matrix)
    print('Sigma for Gaussian kernel chosen as the median observed distance:', 
          sigma)
    thresholded_matrix = threshold_distance_matrix(distance_matrix, threshold)
    adjacency_matrix = compute_similarity_weights(thresholded_matrix, sigma)
    return adjacency_matrix


def plot_distance_distribution(distances, bins=50, save_path=None):
    """Plot histogram of pairwise distances.
    
    Args:
        distances (np.ndarray): Square distance matrix
        bins (int): Number of histogram bins
    """
    dist_values = distances[np.triu_indices_from(distances, k=1)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(dist_values, bins=bins, edgecolor='black')
    plt.title('Distribution of Pairwise Euclidean Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_connectivity_versus_threshold(size_threshold_pairs, save_path=None):
    """Plot relative size of largest connected component versus threshold.
    
    Args:
        size_threshold_pairs (list): List of (relative size, threshold) pairs
    """
    sizes, thresholds = zip(*size_threshold_pairs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sizes, marker='o', color='blue', linestyle='-', linewidth=2)
    plt.title('Connectivity versus Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Relative Size of Largest Connected Component')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    print('Loading data...')
    embeddings = np.load(EMBEDDINGS_PATH)

    print('Computing pairwise distance matrix...')
    X = get_embeddings_subset(embeddings, layer=LAYER)
    distances = squareform(pdist(X, 'euclidean'))

    print('Plotting distance distribution...')
    plot_distance_distribution(distances, save_path=DISTANCE_HISTOGRAM_PATH)

    print('Computing connectivity versus threshold...')
    size_threshold_pairs = get_connectivity_versus_threshold(distances)

    print('Plotting connectivity versus threshold...')
    plot_connectivity_versus_threshold(size_threshold_pairs, 
                                       save_path=CONNECTIVITY_V_THRESHOLD_PATH)
    
    print('Selecting threshold and constructing adjacency matrix...')
    threshold, size = select_threshold(size_threshold_pairs)
    print('Selected threshold:', threshold)
    print('Relative size of largest connected component:', size)

    adjacency_matrix = construct_graph(X, threshold)
    
    print('Saving adjacency matrix...')
    np.save(ADJACENCY_PATH, adjacency_matrix)
