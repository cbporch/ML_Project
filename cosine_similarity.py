import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_pairs
from sklearn.decomposition import PCA

reduction_dim = 500
k_fold = 10

# Training Data
lfw = fetch_lfw_pairs(subset='train')
pairs = lfw.pairs  # 2200 pairs first 1100 are matches, last 1100 are not

# 10-Fold Validation Set
lfw_val = fetch_lfw_pairs(subset='10_folds')
val_set = lfw_val.pairs

# x : vector
# y : vector
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
def cs(x, y, matrix_a):
    # returns a kernel matrix
    return cosine_similarity(np.dot(matrix_a, x), np.dot(matrix_a, y))


def g_a(pos_x, pos_y, neg_x, neg_y, matrix_a, alpha=1):
    return sum(cs(pos_x, pos_y, matrix_a)) - alpha * sum(cs(neg_x, neg_y, matrix_a))


def h_a(matrix_a, beta, matrix_a_zero):
    return beta * np.linalg.norm(matrix_a - matrix_a_zero)


def f_a(pos_x, pos_y, neg_x, neg_y, matrix_a, matrix_a_zero, beta):
    return g_a(pos_x, pos_y, neg_x, neg_y, matrix_a) - h_a(matrix_a, beta, matrix_a_zero)


def cve(t, matrix_a, k_fold=k_fold):  # 10-fold cross validation
    size = len(t)
    # todo - Transform all samples in T using Matrix a

    # Partition T into K equal sized subsamples
    subsamples = []
    step = int(size/k_fold)
    for i in range(k_fold):
        subsamples.append(t[i * step:(i * step) + step])
    total_error = 0
    for k_fold in subsamples:
        # todo - using subsample k as testing data, other K-1 subsamples as training data
        pass
    return total_error/k_fold


def csml(samples, t, d, a):
    # Split into matching (pos) and not matching (neg) pairs
    pos_pairs = dim_red_pairs[:1100]
    neg_pairs = dim_red_pairs[1100:]

    matrix_a_zero = []  # todo: set this
    min_cve = sys.maxint


# ***** Start Pre-processing *****

# Feature Extraction : Intensity
# concatenate all pixels together

new_p = []
for x in pairs:
    new_p.append([np.ravel(x[0]),
                  np.ravel(x[1])])

extracted_pairs = np.array(new_p)


# Dimension Reduction
# using PCA


def reduce_dim(pairs, dim=reduction_dim):
    pca = PCA(n_components=dim)
    pair_1 = pca.fit_transform(pairs[:, 0])
    pair_2 = pca.fit_transform(pairs[:, 1])
    new_pairs = []
    for i in range(len(pairs)):
        new_pairs.append([pair_1[i], pair_2[i]])
    return np.array(new_pairs), pca.get_covariance()


dim_red_pairs, covariance = reduce_dim(extracted_pairs)
print(np.shape(dim_red_pairs))

# Feature Combination?
# todo: SVM for verification, according to the paper

# Choose A_0 - find starting value for A_0, using Whitened PCA
# Whitened PCA is a diagonal matrix(d,m) of the largest eigenvalues of the covariance matrix from PCA

eig = sorted(np.linalg.eigvals(covariance))[::-1]
eig = np.diag(eig[0:reduction_dim])
zero = np.zeros((reduction_dim, len(covariance) - reduction_dim))
A_p = np.concatenate((eig, zero), axis=1)

# ***** End of Pre-processing *****

csml(samples=dim_red_pairs, t=val_set, d=reduction_dim, a=A_p)
