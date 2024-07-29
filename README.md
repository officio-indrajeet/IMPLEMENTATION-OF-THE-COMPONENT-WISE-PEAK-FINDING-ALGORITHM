# IMPLEMENTATION-OF-THE-COMPONENT-WISE-PEAK-FINDING-ALGORITHM


## Overview

This repository contains the implementation of the CPF Clustering Algorithm in two versions:
1. **Existing-CPF.py**: The original implementation of the CPF clustering algorithm.
2. **Optimized-CPF.py**: An optimized version of the CPF clustering algorithm using FAISS for improved performance.

## Dataset

The dataset used for this implementation is the "Turkey Student Evaluation Data Set" available at `turkiye-student-evaluation_generic.csv`.

## Requirements

- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - scipy
  - faiss

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/your-username/CPF-Clustering.git
    cd CPF-Clustering
    ```

2. Install the required Python packages:

    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn scipy faiss-cpu
    ```

## Instructions to Run the Code

### Running Existing-CPF.py

1. Ensure the dataset `turkiye-student-evaluation_generic.csv` is in the same directory as the script.

2. Run the script:

    ```sh
    python Existing-CPF.py
    ```

3. The script will:
   - Load and preprocess the dataset.
   - Perform CPF clustering.
   - Print clustering metrics.
   - Plot PCA results.
   - Print a comparison of the clustering methods.

### Running Optimized-CPF.py

1. Ensure the dataset `turkiye-student-evaluation_generic.csv` is in the same directory as the script.

2. Run the script:

    ```sh
    python Optimized-CPF.py
    ```

3. The script will:
   - Load and preprocess the dataset.
   - Perform CPF clustering using FAISS for optimized performance.
   - Print clustering metrics.
   - Plot PCA results.
   - Print a comparison of the clustering methods.

## Code Explanation

### Existing-CPF.py

This script implements the original CPF clustering algorithm. It consists of three main steps:

1. **Build CCGraph**: Constructs the mutual k-nearest neighbor graph.
2. **Get Density Dists BB**: Computes density distances and big brothers using the mutual k-nearest neighbor graph.
3. **Get Y**: Performs clustering based on density distances and big brothers.

### Optimized-CPF.py

This script implements an optimized version of the CPF clustering algorithm using FAISS. FAISS is used to speed up the nearest neighbor search, improving the performance of the algorithm. It follows the same steps as the original implementation but with FAISS for nearest neighbor search.

## Performance Comparison

The `Optimized-CPF.py` script provides significant performance improvements over the `Existing-CPF.py` script due to the use of FAISS. The table below compares the performance metrics of both versions:

| Method            | Time Taken (seconds) | Number of Clusters | ARI       | AMI       | Silhouette Score | Davies-Bouldin Index |
|-------------------|----------------------|--------------------|-----------|-----------|-------------------|----------------------|
| Original CPF      | 0.78                 | 357                | -0.000744 | -0.001803 | -0.209047         | 1.470275             |
| Optimized CPF     | 0.13                 | 4                  | 0.000000  | 0.000000  | 0.197073          | 0.570039             |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Turkey Student Evaluation Data Set is used in this project.
- FAISS (Facebook AI Similarity Search) library is used for optimizing the CPF clustering algorithm.
