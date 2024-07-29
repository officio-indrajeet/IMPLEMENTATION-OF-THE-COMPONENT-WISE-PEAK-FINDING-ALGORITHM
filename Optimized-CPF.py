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
import faiss

# Optimized CPF functions


def build_CCgraph(X, k, cutoff, n_jobs):
    """
    Step 1: Compute G(X, E), the mutual k-nearest neighbor graph using FAISS for optimization.
    """
    n = X.shape[0]
    index = faiss.IndexFlatL2(X.shape[1])  # Create FAISS index
    index.add(X.astype(np.float32))  # Add data to the index
    # Perform k-nearest neighbor search
    distances, indices = index.search(X.astype(np.float32), k)
    knn_radius = distances[:, k-1]
    # Initialize an empty sparse matrix
    CCmat = scipy.sparse.lil_matrix((n, n))
    for i in range(n):
        for j in indices[i, :]:
            CCmat[i, j] = 1
            CCmat[j, i] = 1
    _, components = scipy.sparse.csgraph.connected_components(
        CCmat, directed=False, return_labels=True)
    comp_labs, comp_count = np.unique(components, return_counts=True)
    outlier_components = comp_labs[comp_count <= cutoff]
    nanidx = np.in1d(components, outlier_components)
    components = components.astype(float)
    if sum(nanidx) > 0:
        components[nanidx] = np.nan
    return components, CCmat, knn_radius


def get_density_dists_bb(X, k, components, knn_radius, n_jobs):
    """
    Step 2: Compute density distances and big brothers using mutual k-nearest neighbor graph with FAISS.
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
        index = faiss.IndexFlatL2(X.shape[1])  # Create FAISS index
        index.add(X[cc_idx, :].astype(np.float32))  # Add data to the index
        distances, neighbors = index.search(X[cc_idx, :].astype(
            np.float32), kcc)  # Perform k-nearest neighbor search
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

        # Step 4: Sort the x's according to their γ values (here, best_distance)
        cc_best_distance = best_distance[cc_idx]
        cc_centers = []

        # Step 5: Let x* = arg max_{x ∈ S} γ(x)
        cc_cut_idx = np.where(knn_radius[cc_idx] >= (
            rho * max(knn_radius[cc_idx])))[0]
        cc_centers.append(cc_cut_idx[np.argmax(cc_best_distance[cc_cut_idx])])

        not_tested = np.ones(nc, dtype=bool)

        while sum(not_tested) > 0:
            # Step 9: Let x* = arg max_{x ∈ S} {γ(x) : x ∉ M}
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

        # Step 17: Initialise G(S, E), a directed graph with S as vertices and no edges, E = ∅
        Clustmat = scipy.sparse.csr_matrix(
            (np.ones((nc)), (BBTree[:, 0], BBTree[:, 1])), shape=(nc, nc))

        # Steps 18-20: for each x in S\M do, Add a directed edge from x to b(x), end for
        # Step 21: for each cluster center x ∈ M do
        n_clusts, cc_y_pred = scipy.sparse.csgraph.connected_components(
            Clustmat, directed=True, return_labels=True)

        # Step 22: Let C be the collection of the points connected by any directed path in G(S, E) that terminates at x.
        # Step 23: Add C ∪ x to Ĉ.
        cc_y_pred += n_cent
        n_cent += n_clusts
        y_pred[cc_idx] = cc_y_pred

    # Step 26: return Ĉ
    return y_pred


class CPFclusterOptimized:
    def __init__(self, k, rho=0.4, alpha=1, n_jobs=-1, remove_duplicates=False, cutoff=1):
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

        # Total time taken
        total_time = step1_time + step2_time + step3_time
        print(f"Total Time taken: {total_time:.4f} seconds")


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

    return valid_indices, ari, ami, silhouette_avg, davies_bouldin


# Use the CPF cluster
cpf_model = CPFclusterOptimized(k=10, rho=0.4, n_jobs=-1)

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
valid_indices, ari_cpf, ami_cpf, silhouette_avg_cpf, davies_bouldin_cpf = print_clustering_metrics(
    true_labels, predicted_labels_cpf)

# Plot PCA results
plot_pca_2d(data_normalized, predicted_labels_cpf,
            'PCA of Student Evaluation Data with CPF Clusters', 'pca_clusters_CPF.png')
