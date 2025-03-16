import numpy as np

def regularized_cca(X, Y, lambda1=0, lambda2=0, n_components=2):
    n_samples = X.shape[0]
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=0)
    Y_std = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0, ddof=0)
    
    Sxx = np.dot(X_std.T, X_std) / (n_samples - 1)
    Syy = np.dot(Y_std.T, Y_std) / (n_samples - 1)
    Sxy = np.dot(X_std.T, Y_std) / (n_samples - 1)
    Syx = Sxy.T

    p, q = Sxx.shape[0], Syy.shape[0]
    Sxx_reg = Sxx + lambda1 * np.eye(p)
    Syy_reg = Syy + lambda2 * np.eye(q)
    
    Ux, sx, _ = np.linalg.svd(Sxx_reg)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(sx))
    Sxx_reg_inv_sqrt = Ux @ D_inv_sqrt @ Ux.T

    Uy, sy, _ = np.linalg.svd(Syy_reg)
    D_inv = np.diag(1.0 / sy)
    Syy_reg_inv = Uy @ D_inv @ Uy.T
    
    M = Sxx_reg_inv_sqrt @ Sxy @ Syy_reg_inv @ Syx @ Sxx_reg_inv_sqrt
    eigvals, eigvecs = np.linalg.eig(M)
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvals = np.real(eigvals[sorted_idx])
    eigvecs = np.real(eigvecs[:, sorted_idx])
    canonical_corr = np.sqrt(np.maximum(eigvals, 0))
    
    A = Sxx_reg_inv_sqrt @ eigvecs
    B = np.zeros((q, n_components))
    for i in range(n_components):
        b = np.linalg.solve(Syy_reg, Syx @ A[:, i])
        B[:, i] = b / canonical_corr[i]
    return canonical_corr[:n_components], A[:, :n_components], B[:, :n_components]

