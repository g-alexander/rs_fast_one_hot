from .rs_fast_one_hot import RsOneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple
from scipy.sparse import csr_matrix

__version__ = "0.1.0"

class OneHotTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_jobs: int = 0):
        super().__init__()
        self.transformer = None
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.transformer = RsOneHotEncoder(X, self.n_jobs)
        return self

    def to_single_thread(self):
        self.n_jobs = 1
        self.transformer.to_single_thread()

    def transform(self, X: List[str]) -> csr_matrix:
        return self.transformer.encode(X)

    def __getstate__(self):
        return {
            'map': self.transformer.get_map(),
            'n_jobs': self.n_jobs
        }

    def __setstate__(self, state):
        self.n_jobs = state['n_jobs']
        self.transformer = RsOneHotEncoder.from_map(state['map'], self.n_jobs)