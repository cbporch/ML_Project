import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_pairs
from sklearn.decomposition import PCA

reduction_dim = 200
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
    s = cosine_similarity(np.dot(matrix_a.T, x).reshape(1, -1), np.dot(matrix_a.T, y).reshape(1, -1))[0][0]
    # print(s)
    return s


# pos_x_slice, pos_y_slice : slices with corresponding pairs that match
# neg_x_slice, neg_y_slice : slices with corresponding pairs that do not match
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# alpha : used to weight the function, set to 1 since len(pos set) = len(neg set)
def g_a(x0, pos_x_slice, pos_y_slice, neg_x_slice, neg_y_slice):
    print("g")
    pos_sum = neg_sum = 0
    for i in range(len(pos_x_slice)):
        pos_sum += cs(pos_x_slice[i], pos_y_slice[i], x0)
        neg_sum += cs(neg_x_slice[i], neg_y_slice[i], x0)
    return pos_sum - neg_sum


# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# beta : weight parameter
# matrix_a_zero : starting value of matrix_a
def h_a(matrix_a, beta, matrix_a_zero):
    print("h")
    return beta * np.linalg.norm(matrix_a - matrix_a_zero)


# pos_x_slice, pos_y_slice : slices with corresponding pairs that match
# neg_x_slice, neg_y_slice : slices with corresponding pairs that do not match
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# beta : weight parameter
# matrix_a_zero : starting value of matrix_a
def f_a(x0, *args):
    x0 = np.reshape(x0, (reduction_dim, 2914))
    pos_pairs, neg_pairs, matrix_a_zero, beta = args

    pos_x_slice = pos_pairs[:, 0]
    pos_y_slice = pos_pairs[:, 1]
    neg_x_slice = neg_pairs[:, 0]
    neg_y_slice = neg_pairs[:, 1]

    g = g_a(x0, pos_x_slice, pos_y_slice, neg_x_slice, neg_y_slice)
    h = h_a(x0, beta, matrix_a_zero)

    print(g)
    print(h)
    return g - h


def sum_gradcs(pairs, a_):
    a_ = np.reshape(a_, (reduction_dim, 2914))
    x_i = pairs[:, 0]
    y_i = pairs[:, 1]

    sum_ = 0
    for i in range(len(x_i)):
        pos_u = np.dot(np.dot(x_i[i].T, a_), np.dot(a_.T, y_i[i]))
        pos_v = np.sqrt(np.dot(np.dot(x_i[i].T, a_), np.dot(a_.T, x_i[i]))) * \
                np.sqrt(np.dot(np.dot(y_i[i].T, a_), np.dot(a_.T, y_i[i])))

        grad_u = np.dot(a_, (np.dot(x_i[i], y_i[i].T) + np.dot(y_i[i], x_i[i].T)))
        grad_v = (np.sqrt(np.dot(np.dot(y_i[i].T, a_), np.dot(a_.T, y_i[i]))) /
                  np.sqrt(np.dot(np.dot(x_i[i].T, a_), np.dot(a_.T, x_i[i])))) * np.dot(a_, np.dot(x_i[i],
                                                                                                   x_i[i].T)) - \
                 (np.sqrt(np.dot(np.dot(x_i[i].T, a_), np.dot(a_.T, x_i[i]))) /
                  np.sqrt(np.dot(np.dot(y_i[i].T, a_), np.dot(a_.T, y_i[i])))) * np.dot(a_, np.dot(y_i[i],
                                                                                                   y_i[i].T))

        sum_ += ((1 / pos_v) * grad_u) - ((pos_u / pos_v ** 2) * grad_v)
    return sum_


def gradf(x0, pos_pairs, neg_pairs, matrix_a_zero, beta):
    x0 = np.reshape(x0, (reduction_dim, 2914))
    return sum_gradcs(pos_pairs, x0) - sum_gradcs(neg_pairs, x0) - (2 * beta * (x0 - matrix_a_zero))


# t : Validation Set
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# k_fold : number of subsets to break t into
def cve(t, matrix_a, k_fold_size=k_fold):  # 10-fold cross validation
    size = len(t)
    for u in range(len(t[:, 0])):
        t[:, 0][u] = np.dot(matrix_a[[range(reduction_dim)],[range(reduction_dim)]], t[:, 0][u])

    # Partition T into K equal sized subsamples
    subsamples = []
    step = int(size / k_fold_size)
    for i in range(k_fold_size):
        subsamples.append(t[i * step:(i * step) + step])
    total_error = 0
    index = 0
    for k_fold in np.array(subsamples):
        # determine threshold
        theta = 0.01  # cosine similarity returns range {-1 to 1}, -1 if dissimilar, 0 if unrelated, 1 if similar
        test_error = 0
        for k in k_fold:
            # get error
            sim = cs(k[0], k[1], matrix_a)
            if val_labels[index] == 1 and sim < theta:
                # false negative
                # print("FN: {3} {2} {0} {1}".format(k[0][0], k[1][0], sim, val_labels[index]))
                test_error += 1
            if val_labels[index] == 0 and sim > theta:
                # false positive
                # print("FP: {3} {2} {0} {1}".format(k[0][0], k[1][0], sim, val_labels[index]))
                test_error += 1
            index += 1
        total_error += test_error / len(k_fold)
        print("k_fold err : {0}".format(test_error / len(k_fold)))
    return total_error / k_fold_size


# samples : Training Data
# t : Validation Set
# d : dimension
# a : starting value for matrix_a
def csml(samples, t, matrix_a_p, d=reduction_dim):
    matrix_a_next = matrix_a_zero = matrix_a_p
    min_cve = curr_cve = float("inf")

    # Split into matching (pos) and not matching (neg) pairs
    pos_pairs = samples[:1100]
    neg_pairs = samples[1100:]

    for n in range(3):
        if min_cve <= 0:
            continue
        for b in range(100, 10, -5):
            if min_cve <= 0:
                continue
            print(b)
            matrix_a_star = gradf(matrix_a_next, pos_pairs, neg_pairs, matrix_a_zero, b)
            matrix_a_star = np.reshape(matrix_a_star, (reduction_dim, 2914))
            curr_cve = cve(t=t, matrix_a=matrix_a_star)
            print("curr_cve: {0}".format(curr_cve))
            if curr_cve < min_cve:
                min_cve = curr_cve
                print("min_cve: {0}".format(curr_cve))
                matrix_a_next = matrix_a_star
        print("min_cve for b={1}: {0}".format(curr_cve,b))
        matrix_a_zero = matrix_a_next
    print("final cve: {0}".format(min_cve))
    return matrix_a_zero


#################################################
# ***** Start Pre-processing *****
# 1.) Feature Extraction : Intensity
# concatenate all pixels together

print("Feature Extraction")
new_training_pairs = []
for x in training_pairs:
    new_training_pairs.append([np.ravel(x[0]),
                               np.ravel(x[1])])
train_pairs = np.array(new_training_pairs)
print(np.shape(train_pairs))

new_val = []
for x in val_set:
    new_val.append([np.ravel(x[0]),
                    np.ravel(x[1])])
val_pairs = np.array(new_val)
print(np.shape(val_pairs))

# 2.) Dimension Reduction
# using PCA
# pairs : pairs of LFW data to transform
# dim: dimension to reduce set to
print('Dimension Reduction')


def reduce_dim(pairs, dim=reduction_dim):
    pca = PCA(n_components=dim)
    size = len(pairs)
    new_pairs = pca.fit_transform(np.concatenate((pairs[:, 0], pairs[:, 1]), axis=0))
    pair1 = new_pairs[:size]
    pair2 = new_pairs[size:]
    new_pairs = np.stack((pair1, pair2), axis=1)
    return np.array(new_pairs), pca.get_covariance()


train_pairs, covariance = reduce_dim(train_pairs)
print(np.shape(train_pairs))

val_pairs, c = reduce_dim(val_pairs)
print(np.shape(val_pairs))

# 3.) Feature Combination?
# todo: SVM for verification, according to the paper

# 4.) Choose A_0 - find starting value for A_0, using Whitened PCA
# Whitened PCA is a diagonal matrix(d,m) of the largest eigenvalues of the covariance matrix from PCA
print("choose A_0")
eig = sorted(np.linalg.eigvals(covariance))[::-1]
eig = np.diag(eig[0:reduction_dim])
zero = np.zeros((reduction_dim, len(covariance) - reduction_dim))
A_p = np.concatenate((eig, zero), axis=1)
print(np.shape(A_p))
# ***** End of Pre-processing *****
#################################################
print("Pre-processing complete")
csml(samples=train_pairs, t=val_pairs, matrix_a_p=A_p)
