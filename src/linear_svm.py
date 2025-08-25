import numpy as np
from cvxopt import matrix, solvers


class LinearSVM:
    def __init__(self, C=1.0, eps=1e-6, Q_jitter=1e-8):
        self.C = float(C)
        self.eps = eps
        self.Q_jitter = Q_jitter
        self.meu = None       # full vector of mu
        self.meu_sv = None    # mu for support vectors
        self.sv = None        # support vectors (X)
        self.sv_y = None      # labels for SVs
        self.w = None
        self.b = 0.0

    # X: numpy array shape (n_samples, n_features)
    # y: numpy array shape (n_samples,) with values {-1, +1}
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        # linear kernel matrix: K = X X^T
        K = X.dot(X.T)

        # Q matrix: Q_ij = y_i y_j K_ij
        Q_np = np.outer(y, y) * K
        # For numerical stability, add tiny jitter to diagonal
        Q_np = Q_np + self.Q_jitter * np.eye(n_samples)

        # Convert to cvxopt matrices
        Q = matrix(Q_np.astype(np.double))
        q = matrix(-np.ones(n_samples, dtype=np.double))  # -1 vector

        # Equality constraint: y^T mu  = 0
        A = matrix(y.reshape(1, -1).astype(np.double))
        b = matrix(0.0)

        # Inequality constraints for box 0 <= mu <= C:
        # G = [ -I ; I ], h = [ 0 ; C*1 ]
        G_np = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h_np = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))

        G = matrix(G_np.astype(np.double))
        h = matrix(h_np.astype(np.double))

        # Solve QP
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, q, G, h, A, b)

        meu = np.ravel(sol['x']).astype(np.float64)

        # Support vectors: mu > eps
        sv_mask = meu > self.eps
        self.meu = meu
        self.meu_sv = meu[sv_mask]
        self.sv = X[sv_mask]
        self.sv_y = y[sv_mask]

        # Compute weight vector w* = sum_i mu_i y_i x_i (only SVs needed)
        self.w = ((self.meu_sv * self.sv_y)[:, None] * self.sv).sum(axis=0)

        # Compute bias b using margin-support vectors: 0 < mu_i < C and w* 
        margin_mask = (meu > self.eps) & (meu < self.C - self.eps)
        if np.any(margin_mask):
            b_vals = y[margin_mask] - X[margin_mask].dot(self.w)
            self.b = np.mean(b_vals)
        else:
            # average over all support vectors
            if self.sv.shape[0] > 0:
                b_vals = self.sv_y - self.sv.dot(self.w)
                self.b = np.mean(b_vals)
            else:
                self.b = 0.0

        return self
    

    # Returns decision function values (w^T x + b)
    def project(self, X):
        return X.dot(self.w) + self.b
    

    # Returns predicted labels in {-1, +1}
    def predict(self, X):
        vals = self.project(X)
        # Use a deterministic threshold to avoid returning 0 for exact boundary cases
        return np.where(vals >= 0.0, 1, -1).astype(int)
    