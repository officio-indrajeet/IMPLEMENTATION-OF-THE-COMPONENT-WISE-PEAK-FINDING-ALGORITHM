import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import time

# CPF functions


def build_CCgraph(X, k, cutoff, n_jobs):
    """
    Step 1: Compute G(X, E), the mutual k-nearest neighbor graph.

    Parameters:
    - X: Input data
    - k: Number of neighbors
    - cutoff: Cutoff for outlier detection
    - n_jobs: Number of parallel jobs

    Returns:
    - components: Connected components of the graph
    - CCmat: Adjacency matrix of the graph
    - knn_radius: Radius of the k-nearest neighbors
    """
    n = X.shape[0]
    # Fit k-nearest neighbors
    kdt = NearestNeighbors(n_neighbors=k, metric='euclidean',
                           n_jobs=n_jobs, algorithm='kd_tree').fit(X)
    # Build the k-neighbors graph
    CCmat = kdt.kneighbors_graph(X, mode='distance')
    distances, _ = kdt.kneighbors(X)
    knn_radius = distances[:, k-1]
    # Symmetrize the graph
    CCmat = CCmat.minimum(CCmat.T)
    # Get connected components
    _, components = scipy.sparse.csgraph.connected_components(
        CCmat, directed=False, return_labels=True)
    # Remove small components (outliers)
    comp_labs, comp_count = np.unique(components, return_counts=True)
    outlier_components = comp_labs[comp_count <= cutoff]
    nanidx = np.in1d(components, outlier_components)
    components = components.astype(float)
    if sum(nanidx) > 0:
        components[nanidx] = np.nan
    return components, CCmat, knn_radius


def get_density_dists_bb(X, k, components, knn_radius, n_jobs):
    """
    Step 2: Compute density distances and big brothers using mutual k-nearest neighbor graph.

    Parameters:
    - X: Input data
    - k: Number of neighbors
    - components: Connected components of the graph
    - knn_radius: Radius of the k-nearest neighbors
    - n_jobs: Number of parallel jobs

    Returns:
    - best_distance: Best density distance for each point
    - big_brother: Big brother for each point
    """
    best_distance = np.empty((X.shape[0]))
    best_distance[:] = np.nan
    big_brother = np.empty((X.shape[0]))
    big_brother[:] = np.nan
    comps = np.unique((components[~np.isnan(components)])).astype(int)

    for cc in comps:
        cc_idx = np.where(components == cc)[0]
        nc = len(cc_idx)
        kcc = min(k, nc-1)
        kdt = NearestNeighbors(n_neighbors=kcc, metric='euclidean',
                               n_jobs=n_jobs, algorithm='kd_tree').fit(X[cc_idx, :])
        distances, neighbors = kdt.kneighbors(X[cc_idx, :])
        cc_knn_radius = knn_radius[cc_idx]
        cc_best_distance = np.empty((nc))
        cc_big_brother = np.empty((nc))

        cc_radius_diff = cc_knn_radius[:,
                                       np.newaxis] - cc_knn_radius[neighbors]
        rows, cols = np.where(cc_radius_diff > 0)
        rows, unidx = np.unique(rows, return_index=True)
        cols = cols[unidx]

        cc_best_distance[rows] = distances[rows, cols]
        cc_big_brother[rows] = neighbors[rows, cols]

        # Ensure indices are within bounds
        cc_big_brother[cc_big_brother >= len(cc_idx)] = len(cc_idx) - 1
        cc_big_brother = cc_idx[cc_big_brother.astype(int)]

        big_brother[cc_idx] = cc_big_brother
        best_distance[cc_idx] = cc_best_distance

    return best_distance, big_brother


def get_y(CCmat, components, knn_radius, best_distance, big_brother, rho, alpha, d):
    """
    Steps 3-26: Perform clustering based on density distances and big brothers.

    Parameters:
    - CCmat: Adjacency matrix of the graph
    - components: Connected components of the graph
    - knn_radius: Radius of the k-nearest neighbors
    - best_distance: Best density distance for each point
    - big_brother: Big brother for each point
    - rho: Density threshold
    - alpha: Alpha parameter for clustering
    - d: Dimension of the data

    Returns:
    - y_pred: Predicted cluster labels
    """
    y_pred = np.empty((CCmat.shape[0]))
    y_pred[:] = np.nan
    n_cent = 0
    peaks = []
    comps = np.unique((components[~np.isnan(components)])).astype(int)

    for cc in comps:
        cc_idx = np.where(components == cc)[0]
        nc = len(cc_idx)
        if nc <= 2:
            y_pred[cc_idx] = n_cent
            n_cent += 1
            continue

        # Sort the points according to their best distances
        cc_best_distance = best_distance[cc_idx]
        cc_centers = []

        # Find the initial center
        cc_cut_idx = np.where(knn_radius[cc_idx] >= (
            rho * max(knn_radius[cc_idx])))[0]
        cc_centers.append(cc_cut_idx[np.argmax(cc_best_distance[cc_cut_idx])])

        not_tested = np.ones(nc, dtype=bool)

        while sum(not_tested) > 0:
            prop_cent = np.argmax(cc_best_distance * not_tested)
            if prop_cent not in cc_centers:
                cc_centers.append(prop_cent)
            not_tested[prop_cent] = False

            if len(cc_centers) > 1:
                min_knn_radius_center = np.argmin(knn_radius[cc_centers])
                if knn_radius[prop_cent] == knn_radius[cc_centers[min_knn_radius_center]]:
                    break

        cc_centers = np.array(cc_centers)
        peaks.extend(cc_idx[cc_centers])
        BBTree = np.zeros((nc, 2))
        BBTree[:, 0] = range(nc)
        BBTree[:, 1] = big_brother[cc_idx]
        BBTree[cc_centers, 1] = cc_centers
        BBTree = BBTree.astype(int)

        # Ensure indices are within bounds
        BBTree[BBTree[:, 1] >= nc, 1] = nc - 1

        # Initialize the directed graph
        Clustmat = scipy.sparse.csr_matrix(
            (np.ones((nc)), (BBTree[:, 0], BBTree[:, 1])), shape=(nc, nc))

        # Perform clustering
        n_clusts, cc_y_pred = scipy.sparse.csgraph.connected_components(
            Clustmat, directed=True, return_labels=True)

        cc_y_pred += n_cent
        n_cent += n_clusts
        y_pred[cc_idx] = cc_y_pred

    return y_pred


class CPFcluster:
    def __init__(self, k, rho=0.4, alpha=1, n_jobs=1, remove_duplicates=False, cutoff=1):
        self.k = k
        self.rho = rho
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.remove_duplicates = remove_duplicates
        self.cutoff = cutoff

    def fit(self, X):
        """
        Fit the CPF model to the data.
        """
        if self.remove_duplicates:
            X = np.unique(X, axis=0)

        n, d = X.shape
        if self.k > n:
            raise ValueError("k cannot be larger than n.")

        # Step 1: Build CCGraph
        start_time = time.time()
        components, CCmat, knn_radius = build_CCgraph(
            X, self.k, self.cutoff, self.n_jobs)
        step1_time = time.time() - start_time
        print(f"Step 1: Build CCGraph took {step1_time:.4f} seconds")

        # Step 2: Get Density Dists BB
        start_time = time.time()
        best_distance, big_brother = get_density_dists_bb(
            X, self.k, components, knn_radius, self.n_jobs)
        step2_time = time.time() - start_time
        print(f"Step 2: Get Density Dists BB took {step2_time:.4f} seconds")

        # Step 3: Get Y
        start_time = time.time()
        self.labels_ = get_y(CCmat, components, knn_radius,
                             best_distance, big_brother, self.rho, self.alpha, d)
        step3_time = time.time() - start_time
        print(f"Step 3: Get Y took {step3_time:.4f} seconds")


# Load the dataset
file_path = 'turkiye-student-evaluation_generic.csv'
data = pd.read_csv(file_path)

# Define features and remove non-Likert scale attributes
features = data.drop(
    columns=['instr', 'class', 'nb.repeat', 'attendance', 'difficulty'])

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(features)

# Add synthetic labels for ARI and AMI calculations
np.random.seed(42)
# Assuming 3 clusters for synthetic labels
true_labels = np.random.randint(0, 3, size=data_normalized.shape[0])

# Function to plot PCA results


def plot_pca_2d(data, labels, title, file_name):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1],
                    hue=labels, palette='tab10', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.savefig(file_name)
    plt.show()

# Function to calculate and print clustering metrics


def print_clustering_metrics(true_labels, predicted_labels):
    # Removing NaN entries
    valid_indices = ~np.isnan(predicted_labels)
    true_labels = true_labels[valid_indices]
    predicted_labels = predicted_labels[valid_indices]

    ari = adjusted_rand_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    silhouette_avg = silhouette_score(
        data_normalized[valid_indices], predicted_labels)
    davies_bouldin = davies_bouldin_score(
        data_normalized[valid_indices], predicted_labels)

    print(f'Adjusted Rand Index (ARI): {ari:.6f}')
    print(f'Adjusted Mutual Information (AMI): {ami:.6f}')
    print(f'Silhouette Score: {silhouette_avg:.6f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.6f}')


# Use the CPF cluster
cpf_model = CPFcluster(k=10, rho=0.4, n_jobs=-1)

# Measure time taken
start_time = time.time()
cpf_model.fit(data_normalized)
end_time = time.time()

predicted_labels_cpf = cpf_model.labels_

# Calculate number of clusters found
num_clusters = len(
    np.unique(predicted_labels_cpf[~np.isnan(predicted_labels_cpf)]))

print("CPF Clustering:")
print(f"Number of clusters found: {num_clusters}")
print(f"Total Time taken: {end_time - start_time:.2f} seconds")

# Calculate and print clustering metrics
print_clustering_metrics(true_labels, predicted_labels_cpf)

# Plot PCA results
plot_pca_2d(data_normalized, predicted_labels_cpf,
            'PCA of Student Evaluation Data with CPF Clusters', 'pca_clusters_CPF.png')
