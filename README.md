# CPF clustering with Faiss

Python implementation of **Component-wise Peak-Finding (CPF)**, a density-based clustering algorithm, with
interchangeable exact k-NN backends: scikit-learn, [Faiss](https://github.com/facebookresearch/faiss) and NumPy.
Built from my MSc dissertation at Trinity College Dublin.

Original paper: [A Theoretical Analysis of Density Peaks Clustering and the Component-wise Peak-Finding Algorithm](https://www.tara.tcd.ie/items/bbde36f5-a6e4-4f26-b611-f16aff652732)
(Tobin & Zhang, IEEE TPAMI 2024).

## Install

```bash
git clone https://github.com/officio-indrajeet/IMPLEMENTATION-OF-THE-COMPONENT-WISE-PEAK-FINDING-ALGORITHM.git
cd IMPLEMENTATION-OF-THE-COMPONENT-WISE-PEAK-FINDING-ALGORITHM
pip install -r requirements.txt
pip install faiss-cpu   # optional
```

## Usage

```python
from cpf import CPFcluster

model = CPFcluster(k=10, rho=0.4, backend="auto")  # Faiss if installed, else scikit-learn
labels = model.fit_predict(X)                       # -1 = noise
```

## Benchmark and tests

```bash
python benchmark.py                          # nine UCI datasets, results in results/
python -m unittest discover -s tests -v
```

## More

- [Parameters, datasets and caveats](docs/DETAILS.md)
- [What was fixed from the dissertation scripts](docs/CHANGES.md)

## Citation

Tobin, J. and Zhang, M., *A Theoretical Analysis of Density Peaks Clustering and the Component-wise
Peak-Finding Algorithm*, IEEE TPAMI 46(2):1109–1120, 2024.

## License

MIT — see [LICENSE](LICENSE).
