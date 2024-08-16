import numpy as np
import faiss

# Create a random dataset with 1000 samples and 128 features
np.random.seed(42)
data = np.random.random((1000, 128)).astype('float32')

# Number of nearest neighbors to search for
k = 5

# Step 1: Initialize a FAISS index (using L2 distance)
index = faiss.IndexFlatL2(data.shape[1])  # dimension should match the data

# Step 2: Add data to the index
index.add(data)  # FAISS stores data in the index

# Step 3: Perform a search to find the k-nearest neighbors for each point in the dataset
distances, indices = index.search(data, k)

# Output the results
print("Indices of nearest neighbors:")
print(indices[:5])  # Display indices of the nearest neighbors for the first 5 points

print("\nDistances to nearest neighbors:")
print(distances[:5])  # Display distances to the nearest neighbors for the first 5 points
