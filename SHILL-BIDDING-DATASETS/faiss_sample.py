import faiss
import numpy as np

# Generate some random data
d = 64  # dimension
nb = 10000  # database size
np.random.seed(1234)  # make reproducible
data = np.random.random((nb, d)).astype('float32')

# Initialize the FAISS index
index = faiss.IndexFlatL2(d)  # L2 distance
index.add(data)  # add vectors to the index

# Search the nearest neighbors
k = 4  # we want to see 4 nearest neighbors
distances, indices = index.search(data[:5], k)  # actual search
print(indices)
print(distances)
