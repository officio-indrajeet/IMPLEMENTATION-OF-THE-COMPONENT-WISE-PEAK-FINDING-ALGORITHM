import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances
import hdbscan
import time


# Load the dataset
file_path = 'Shill_Bidding_Dataset.csv'
data = pd.read_csv(file_path)

# Identify non-numeric columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# Drop the non-numeric columns
data = data.drop(columns=non_numeric_columns)

# Drop any remaining rows with NaN values
data = data.dropna()

# Reduce dataset size for testing
# data_sampled = data.sample(n=1000, random_state=42)

# Define features and remove target attribute
features = data.drop(columns=['Class'])

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
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=labels, palette='tab10', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.savefig(file_name)
    plt.show()


# Function to calculate and print clustering metrics
# def print_clustering_metrics(true_labels, predicted_labels):
#     ari = adjusted_rand_score(true_labels, predicted_labels)
#     ami = adjusted_mutual_info_score(true_labels, predicted_labels)
#     silhouette_avg = silhouette_score(data_normalized, predicted_labels)
#     davies_bouldin = davies_bouldin_score(data_normalized, predicted_labels)
#
#     print(f'Adjusted Rand Index (ARI): {ari:.6f}')
#     print(f'Adjusted Mutual Information (AMI): {ami:.6f}')
#     print(f'Silhouette Score: {silhouette_avg:.6f}')
#     print(f'Davies-Bouldin Index: {davies_bouldin:.6f}')

def print_clustering_metrics(true_labels, predicted_labels):
    # Check the number of unique clusters
    num_clusters = len(np.unique(predicted_labels))

    if num_clusters == 1:
        print("Only one cluster found. Silhouette Score and Davies-Bouldin Index are not applicable.")
        return

    ari = adjusted_rand_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    silhouette_avg = silhouette_score(data_normalized, predicted_labels)
    davies_bouldin = davies_bouldin_score(data_normalized, predicted_labels)

    print(f'Adjusted Rand Index (ARI): {ari:.6f}')
    print(f'Adjusted Mutual Information (AMI): {ami:.6f}')
    print(f'Silhouette Score: {silhouette_avg:.6f}')
    print(f'Davies-Bouldin Index: {davies_bouldin:.6f}')


# DBSCAN
start_time = time.time()
dbscan = DBSCAN(eps=0.3, min_samples=5)
predicted_labels_dbscan = dbscan.fit_predict(data_normalized)
end_time = time.time()
dbscan_clusters = len(set(predicted_labels_dbscan)) - (1 if -1 in predicted_labels_dbscan else 0)
print("DBSCAN Clustering:")
print(f'Total clusters found (excluding noise): {dbscan_clusters}')
print_clustering_metrics(true_labels, predicted_labels_dbscan)
print(f'Time taken: {end_time - start_time} seconds')
# plot_pca_2d(data_normalized, predicted_labels_dbscan, 'PCA of Shill Bidding Data with DBSCAN Clusters', 'pca_clusters_DBSCAN.png')

# Mean Shift
start_time = time.time()
mean_shift = MeanShift()
predicted_labels_meanshift = mean_shift.fit_predict(data_normalized)
end_time = time.time()
mean_shift_clusters = len(np.unique(predicted_labels_meanshift))
print("\nMean Shift Clustering:")
print(f'Total clusters found: {mean_shift_clusters}')
print_clustering_metrics(true_labels, predicted_labels_meanshift)
print(f'Time taken: {end_time - start_time} seconds')
# plot_pca_2d(data_normalized, predicted_labels_meanshift, 'PCA of Shill Bidding Data with Mean Shift Clusters', 'pca_clusters_MeanShift.png')

# HDBSCAN
start_time = time.time()
hdbscan_clustering = hdbscan.HDBSCAN(min_cluster_size=15)
predicted_labels_hdbscan = hdbscan_clustering.fit_predict(data_normalized)
end_time = time.time()
hdbscan_clusters = len(set(predicted_labels_hdbscan)) - (1 if -1 in predicted_labels_hdbscan else 0)
print("\nHDBSCAN Clustering:")
print(f'Total clusters found (excluding noise): {hdbscan_clusters}')
print_clustering_metrics(true_labels, predicted_labels_hdbscan)
print(f'Time taken: {end_time - start_time} seconds')
# plot_pca_2d(data_normalized, predicted_labels_hdbscan, 'PCA of Shill Bidding Data with HDBSCAN Clusters', 'pca_clusters_HDBSCAN.png')

# QuickShift Implementation


def compute_distance_matrix(data, metric):
    return pairwise_distances(data, metric=metric)


def compute_weight_matrix(dist_matrix, window_type, bandwidth):
    if window_type == 'flat':
        weight_matrix = 1 * (dist_matrix <= bandwidth)
    elif window_type == 'normal':
        weight_matrix = np.exp(-dist_matrix ** 2 / (2 * bandwidth ** 2))
    else:
        raise ValueError("Unknown window type")
    return weight_matrix


def compute_medoids(dist_matrix, weight_matrix, tau):
    P = sum(weight_matrix)
    P = P[:, np.newaxis] - P
    dist_matrix[dist_matrix == 0] = tau / 2
    S = np.sign(P) * (1 / dist_matrix)
    S[dist_matrix > tau] = -1
    return np.argmax(S, axis=0)


def compute_stationary_medoids(data, tau, window_type, bandwidth, metric):
    dist_matrix = compute_distance_matrix(data, metric)
    weight_matrix = compute_weight_matrix(dist_matrix, window_type, bandwidth)
    medoids = compute_medoids(dist_matrix, weight_matrix, tau)
    stationary_idx = [i for i in range(len(medoids)) if medoids[i] == i]
    return medoids, np.asarray(stationary_idx)


def quick_shift(data, tau, window_type, bandwidth, metric):
    if tau is None:
        tau = estimate_bandwidth(data)
    if bandwidth is None:
        bandwidth = estimate_bandwidth(data)

    medoids, cluster_centers_idx = compute_stationary_medoids(data, tau, window_type, bandwidth, metric)
    cluster_centers = data[cluster_centers_idx]
    labels = []
    labels_val = {}
    lab = 0
    for i in cluster_centers_idx:
        labels_val[i] = lab
        lab += 1
    for i in range(len(data)):
        next_med = medoids[i]
        while next_med not in cluster_centers_idx:
            next_med = medoids[next_med]
        labels.append(labels_val[next_med])
    return cluster_centers, np.asarray(labels), cluster_centers_idx


class QuickShift:
    def __init__(self, tau=None, bandwidth=None, window_type="flat", metric="euclidean"):
        self.tau = tau
        self.bandwidth = bandwidth
        self.window_type = window_type
        self.metric = metric

    def fit(self, data):
        self.cluster_centers_, self.labels_, self.cluster_centers_idx_ = quick_shift(
            data, tau=self.tau, window_type=self.window_type, bandwidth=self.bandwidth, metric=self.metric)
        return self


# QuickShift Clustering
start_time = time.time()
quickshift = QuickShift(tau=0.3, bandwidth=None, window_type="flat", metric="euclidean")
quickshift.fit(data_normalized)
predicted_labels_quickshift = quickshift.labels_
end_time = time.time()
quickshift_clusters = len(np.unique(predicted_labels_quickshift))
print("\nQuick Shift Clustering:")
print(f'Total clusters found: {quickshift_clusters}')
print_clustering_metrics(true_labels, predicted_labels_quickshift)
print(f'Time taken: {end_time - start_time} seconds')
# plot_pca_2d(data_normalized, predicted_labels_quickshift, 'PCA of Shill Bidding Data with QuickShift Clusters', 'pca_clusters_QuickShift.png')
