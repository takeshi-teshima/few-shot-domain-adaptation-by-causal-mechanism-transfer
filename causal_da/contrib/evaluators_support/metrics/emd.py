import numpy as np


class EMD:
    def __init__(self):
        import ot
        self.ot = ot

    def __call__(self, A, B):
        M = self.ot.dist(A, B)
        # TODO Make sure this normalization is justified.
        #M /= M.max()
        M /= M.sum()
        a, b = np.ones((len(A), )) / len(A), np.ones((len(B), )) / len(B)
        G0 = self.ot.emd2(a, b, M)
        return G0
