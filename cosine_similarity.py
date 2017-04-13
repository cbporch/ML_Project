import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_pairs
from sklearn.decomposition import PCA
from scipy.optimize import fmin_cg

reduction_dim = 500
k_fold = 10

# Training Data
lfw = fetch_lfw_pairs(subset='train')
training_pairs = lfw.pairs  # 2200 pairs first 1100 are matches, last 1100 are not

# 10-Fold Validation Set
lfw_val = fetch_lfw_pairs(subset='10_folds')
val_set = lfw_val.pairs
val_labels = lfw_val.target


# x : vector
# y : vector
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
def cs(x, y, matrix_a):
    # returns a kernel matrix
    return cosine_similarity(np.dot(matrix_a.T, x).reshape(1,-1), np.dot(matrix_a.T, y).reshape(1,-1))[0][0]


# pos_x_slice, pos_y_slice : slices with corresponding pairs that match
# neg_x_slice, neg_y_slice : slices with corresponding pairs that do not match
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# alpha : used to weight the function, set to 1 since len(pos set) = len(neg set)
def g_a(pos_x_slice, pos_y_slice, neg_x_slice, neg_y_slice, matrix_a, alpha=1):
    pos_sum = neg_sum = 0
    for i in range(len(pos_x_slice)):
        pos_sum += cs(pos_x_slice[i], pos_y_slice[i], matrix_a)
        neg_sum += cs(neg_x_slice[i], neg_y_slice[i], matrix_a)
    return pos_sum - (alpha * neg_sum)


# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# beta : weight parameter
# matrix_a_zero : starting value of matrix_a
def h_a(matrix_a, beta, matrix_a_zero):
    return beta * np.linalg.norm(matrix_a - matrix_a_zero)


# pos_x_slice, pos_y_slice : slices with corresponding pairs that match
# neg_x_slice, neg_y_slice : slices with corresponding pairs that do not match
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# beta : weight parameter
# matrix_a_zero : starting value of matrix_a
def f_a(x0, *args):
    print("f")
    matrix_a = x0
    matrix_a = np.reshape(matrix_a, (500, 2914))
    pos_pairs, neg_pairs, matrix_a_zero, beta = args

    pos_x_slice = pos_pairs[:, 0]
    pos_y_slice = pos_pairs[:, 1]
    neg_x_slice = pos_pairs[:, 0]
    neg_y_slice = pos_pairs[:, 1]

    return g_a(pos_x_slice, pos_y_slice, neg_x_slice, neg_y_slice, matrix_a) - h_a(matrix_a, beta, matrix_a_zero)


# t : Validation Set
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# k_fold : number of subsets to break t into
def cve(t, matrix_a, k_fold=k_fold):  # 10-fold cross validation
    size = len(t)
    for u in range(len(t[:, 0])):
        t[:,0][u] = np.dot(matrix_a, t[:,0][u])

    # Partition T into K equal sized subsamples
    subsamples = []
    step = int(size/k_fold)
    for i in range(k_fold):
        subsamples.append(t[i * step:(i * step) + step])
    total_error = 0
    index = 0
    for k_fold in subsamples:
        # todo - use subsample k as testing data, other K-1 subsamples as training data
        # determine threshold
        theta = 0.7
        test_error = 0
        for k in k_fold:
            # get error
            if val_labels[index] == 1 and cs(k[:, 0], k[:, 1]) < theta:
                # false negative
                test_error += 1
            if val_labels[index] == 0 and cs(k[:, 0], k[:, 1]) > theta:
                # false positive
                test_error += 1
            index += 1
        total_error += test_error/len(k_fold)
    return total_error/k_fold


# samples : Training Data
# t : Validation Set
# d : dimension
# a : starting value for matrix_a
def csml(samples, t, matrix_a_p, d=reduction_dim):
    matrix_a_next = matrix_a_zero = matrix_a_p
    min_cve = float("inf")

    # Split into matching (pos) and not matching (neg) pairs
    pos_pairs = samples[:1100]
    neg_pairs = samples[1100:]

    for n in range(100):
        for b in range(10):
            print(b)
            x0 = matrix_a_zero
            args = (pos_pairs, neg_pairs, matrix_a_next, b)
            matrix_a_star, info = fmin_cg(f=f_a, x0=x0, args=args, maxiter=1)
            matrix_a_star = np.reshape(matrix_a_star, (500, 2914))
            print(np.shape(matrix_a_star))
            curr_cve = cve(t=t, matrix_a=matrix_a_star)
            if curr_cve < min_cve:
                min_cve = curr_cve
                matrix_a_next = matrix_a_star
        matrix_a_zero = matrix_a_next


#################################################
# ***** Start Pre-processing *****
# 1.) Feature Extraction : Intensity
# concatenate all pixels together

new_training_pairs = []
for x in training_pairs:
    new_training_pairs.append([np.ravel(x[0]),
                               np.ravel(x[1])])
extract_pairs = np.array(new_training_pairs)

new_val = []
for x in val_set:
    new_val.append([np.ravel(x[0]),
                    np.ravel(x[1])])
extract_val_pairs = np.array(new_val)


# 2.) Dimension Reduction
# using PCA
# pairs : pairs of LFW data to transform
# dim: dimension to reduce set to
def reduce_dim(pairs, dim=reduction_dim):
    pca = PCA(n_components=dim)
    size = len(pairs)
    new_pairs = pca.fit_transform(np.concatenate((pairs[:, 0], pairs[:, 1]), axis=0))
    pair1 = new_pairs[:size]
    pair2 = new_pairs[size:]
    new_pairs = np.stack((pair1, pair2), axis=1)
    return np.array(new_pairs), pca.get_covariance()

dim_red_pairs, covariance = reduce_dim(extract_pairs)
dim_red_val, c = reduce_dim(extract_val_pairs)

print(np.shape(dim_red_val))

# 3.) Feature Combination?
# todo: SVM for verification, according to the paper

# 4.) Choose A_0 - find starting value for A_0, using Whitened PCA
# Whitened PCA is a diagonal matrix(d,m) of the largest eigenvalues of the covariance matrix from PCA

eig = sorted(np.linalg.eigvals(covariance))[::-1]
eig = np.diag(eig[0:reduction_dim])
zero = np.zeros((reduction_dim, len(covariance) - reduction_dim))
A_p = np.concatenate((eig, zero), axis=1)

# ***** End of Pre-processing *****
#################################################

csml(samples=dim_red_pairs, t=val_set, matrix_a_p=A_p)
