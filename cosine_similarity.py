import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_pairs
from sklearn.datasets import olivetti_faces
from sklearn.decomposition import PCA
from sklearn import svm

import nearest_neighbor

DIM_M = 500
DIM_D = 200
k_fold = 10
is_lfw = True


def build_lfw():
    global is_lfw
    is_lfw = True
    # Training Data
    lfw = fetch_lfw_pairs(subset='train')
    t_pairs = lfw.pairs  # 2200 pairs first 1100 are matches, last 1100 are not

    # 10-Fold Validation Set
    lfw_val = fetch_lfw_pairs(subset='10_folds')
    v_set = lfw_val.pairs
    v_labels = lfw_val.target
    return t_pairs, v_set, v_labels


def build_olivetti():
    global is_lfw
    is_lfw = False
    # fetch regular data
    olivetti = olivetti_faces.fetch_olivetti_faces()
    ol_f_data = olivetti.data
    ol_f_targets = olivetti.target

    # fetch shuffled data
    ol_scramble = olivetti_faces.fetch_olivetti_faces(shuffle=True)
    ol_scram = ol_scramble.data
    ol_scram_target = ol_scramble.target

    pairs = np.stack((ol_f_data, ol_scram), axis=1)
    labels = []
    for i in range(len(pairs)):
        if ol_f_targets[i] == ol_scram_target[i]:
            labels.append(1)
        else:
            labels.append(0)
    rev = np.fliplr(ol_scram)
    next_pairs = np.stack((ol_f_data, rev), axis=1)
    scram_targ_flip = ol_scram_target[::-1]
    for i in range(len(next_pairs)):
        if ol_f_targets[i] == scram_targ_flip[i]:
            labels.append(1)
        else:
            labels.append(0)
    pairs = np.concatenate((pairs, next_pairs))

    matched_pair_set = []
    for i in range(40):
        # for each participant in the dataset, to ensure accurately paired faces
        participant_set = ol_f_data[i*10:(i*10)+10]
        # rot2 and pair
        shifted = np.concatenate(np.array((participant_set[2:], participant_set[:2])))
        shift_stack = np.stack((participant_set, shifted), axis=1)
        # reverse and pair
        flip_stack = np.stack((participant_set, participant_set[::-1]), axis=1)
        # this should produce 20 unique pairings for each participant
        for p, k in zip(flip_stack, shift_stack):
            matched_pair_set.append(p)
            matched_pair_set.append(k)
            labels.append(1)
            labels.append(1)

    matched_pair_set = np.array(matched_pair_set)
    pairs = np.concatenate((pairs, matched_pair_set))

    # todo - build more pairs, split off validation set

    return pairs, labels


# x : vector
# y : vector
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
def cs(x, y, matrix_a):
    # returns a kernel matrix
    # s = cosine_similarity(np.dot(matrix_a, x).reshape(1, -1), np.dot(matrix_a, y).reshape(1, -1))[0][0]
    s = np.dot(np.dot(matrix_a, x), np.dot(matrix_a, y)) / np.dot(
        (np.sqrt(np.dot(np.dot(matrix_a, x), np.dot(matrix_a, x)))),
        np.dot(np.dot(matrix_a, y), np.dot(matrix_a, y)))
    # print(s)
    return s


# pos_x_slice, pos_y_slice : slices with corresponding pairs that match
# neg_x_slice, neg_y_slice : slices with corresponding pairs that do not match
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# alpha : used to weight the function, set to 1 since len(pos set) = len(neg set)
def g_a(x0, pos_x_slice, pos_y_slice, neg_x_slice, neg_y_slice):
    pos_sum = neg_sum = 0
    for i in range(len(pos_x_slice)):
        pos_sum += cs(pos_x_slice[i], pos_y_slice[i], x0)
        neg_sum += cs(neg_x_slice[i], neg_y_slice[i], x0)
    return pos_sum - neg_sum


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
    # x0 = np.reshape(x0, (DIM_D, DIM_M))
    pos_pairs, neg_pairs, matrix_a_zero, beta = args

    pos_x_slice = pos_pairs[:, 0]
    pos_y_slice = pos_pairs[:, 1]
    neg_x_slice = neg_pairs[:, 0]
    neg_y_slice = neg_pairs[:, 1]

    g = g_a(x0, pos_x_slice, pos_y_slice, neg_x_slice, neg_y_slice)
    h = h_a(x0, beta, matrix_a_zero)

    return g - h


def sum_gradcs(pairs, a_):
    # a_ = np.reshape(a_, (DIM_D, DIM_M))
    x_i = pairs[:, 0]
    y_i = pairs[:, 1]
    sum_ = 0
    for i in range(len(x_i)):
        pos_u = np.dot(np.dot(a_, x_i[i]).T, np.dot(a_, y_i[i]))
        pos_v = np.sqrt(np.dot(np.dot(a_, x_i[i]).T, np.dot(a_, x_i[i]))) * \
                np.sqrt(np.dot(np.dot(a_, y_i[i]).T, np.dot(a_, y_i[i])))
        grad_u = np.dot(a_, (np.dot(x_i[i], y_i[i].T) + np.dot(y_i[i], x_i[i].T)))
        grad_v = (np.sqrt(np.dot(np.dot(a_, y_i[i]).T, np.dot(a_, y_i[i]))) /
                  np.sqrt(np.dot(np.dot(a_, x_i[i]).T, np.dot(a_, x_i[i])))) * np.dot(a_, np.dot(x_i[i],
                                                                                                 x_i[i].T)) - \
                 (np.sqrt(np.dot(np.dot(a_, x_i[i]).T, np.dot(a_, x_i[i]))) /
                  np.sqrt(np.dot(np.dot(a_, y_i[i]).T, np.dot(a_, y_i[i])))) * np.dot(a_, np.dot(y_i[i],
                                                                                                 y_i[i].T))
        sum_ += (grad_u / pos_v) - ((pos_u / pos_v ** 2) * grad_v)
    return sum_


def gradf(m_a, pos_pairs, neg_pairs, matrix_a_zero, beta):
    # x0 = np.reshape(x0, (DIM_D, DIM_M))
    pos_sum = sum_gradcs(pos_pairs, m_a)
    neg_sum = sum_gradcs(neg_pairs, m_a)
    # print(pos_sum)
    # print(neg_sum)
    return pos_sum - neg_sum - (2 * beta * (m_a - matrix_a_zero))


def subsamples(t, k_fold_size=k_fold):
    # Partition T into K equal sized subsamples
    size = len(t)
    subsamples = []
    step = int(size / k_fold_size)
    for i in range(k_fold_size):
        subsamples.append(t[i * step:(i * step) + step])
    return np.array(subsamples), size, step


# t : Validation Set
# matrix_a : linear transformation A: R^m -> R^d(d<=m)
# k_fold : number of subsets to break t into
def cve(subsamples, matrix_a, size, step, v_labels, k_fold_size=k_fold):  # 10-fold cross validation
    total_error = 0
    index = 0
    sample = 0

    for k_fold in subsamples:
        # determine threshold
        test_error = 0

        train_k = np.concatenate((subsamples[:sample], subsamples[sample + 1:]), axis=0)
        t_k = []
        for sub in train_k:
            for z in sub:
                t_k.append(z)

        t_k = np.array(t_k)
        train_k_labels = np.concatenate((v_labels[:sample * step], v_labels[((sample * step) + step):]))

        sim_scores = []
        for j in t_k:
            sim_scores.append(cs(j[0], j[1], matrix_a))
        sim_scores = np.array(sim_scores)

        sup_vec_mac = svm.SVC(kernel='linear', degree=2)
        sup_vec_mac.fit(np.reshape(sim_scores, (size - step, 1)), train_k_labels)
        theta = -sup_vec_mac.intercept_[0] / sup_vec_mac.coef_[0]  # separator for k-1 training data

        # theta = nearest_neighbor.getboundry(sim_scores, train_k_labels)
        # nearest_neighbor.plot()
        # print(theta)

        for k in range(len(k_fold)):
            # get error
            sim = cs(k_fold[k][0], k_fold[k][1], matrix_a)
            if v_labels[index] == 1 and sim < theta:
                # false negative
                # print("FN: {3} {2} {0} {1}".format(k[0][0], k[1][0], sim, val_labels[index]))
                test_error += 1
            if v_labels[index] == 0 and sim > theta:
                # false positive
                # print("FP: {3} {2} {0} {1}".format(k[0][0], k[1][0], sim, val_labels[index]))
                test_error += 1
            index += 1
        total_error += test_error / len(k_fold)
        # print("k_fold err : {0}".format(test_error / len(k_fold)))
        sample += 1
    return total_error / k_fold_size


# samples : Training Data
# t : Validation Set
# d : dimension
# a : starting value for matrix_a
def csml(pos_samples, neg_samples, t, matrix_a_p, v):
    matrix_a_next = matrix_a_zero = matrix_a_p
    min_cve = float("inf")

    t, size, step = subsamples(t)
    best_b = 0
    for n in range(5):
        if min_cve <= 0:
            print("final cve: {0}".format(min_cve))
            return matrix_a_zero

        for beta in np.arange(1000, 1, -100):

            matrix_a_star = gradf(matrix_a_zero, pos_samples, neg_samples, matrix_a_p, beta)

            print("f : {0}".format(f_a(matrix_a_zero, pos_samples, neg_samples, matrix_a_p, beta)))
            curr_cve = cve(subsamples=t, size=size, step=step, matrix_a=matrix_a_star, v_labels=v)

            print(max(matrix_a_star))
            # print(matrix_a_star == matrix_a_next)
            if curr_cve < min_cve:
                min_cve = curr_cve
                matrix_a_next = matrix_a_star
                best_b = beta
            print("min_cve for b = {1:2.2f}: {0:1.5f}".format(min_cve, beta))
            matrix_a_zero = matrix_a_next
    print("final cve for beta of {1:2.2f}: {0}".format(min_cve, best_b))
    return matrix_a_zero


#################################################
# ***** Start Pre-processing *****
# 1.) Feature Extraction : Intensity
# concatenate all pixels together

training_pairs, val_set, val_labels = build_lfw()

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


def reduce_dim(pairs, dim=DIM_M):
    pca = PCA(n_components=dim, whiten=True)
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

# 4.) Choose A_0 - find starting value for A_0, using Whitened PCA
# Whitened PCA is a diagonal matrix(d,m) of the largest eigenvalues of the covariance matrix from PCA
print("choose A_0")

eig = sorted(np.linalg.eigvals(covariance))[::-1]
new_eig = []
for e in eig:
    new_eig.append(e ** -0.5)
new_eig = np.diag(new_eig[0:DIM_D])

zero = np.zeros((DIM_D, DIM_M - DIM_D))
A_p = np.concatenate((new_eig, zero), axis=1)

print("A_p shape: {0}".format(np.shape(A_p)))

# ***** End of Pre-processing *****
#################################################
print("Pre-processing complete")

# Split into matching (pos) and not matching (neg) pairs
if is_lfw:
    pos_pairs = train_pairs[:1100]
    neg_pairs = train_pairs[1100:]

csml(pos_samples=pos_pairs, neg_samples=neg_pairs, t=val_pairs, v=val_labels, matrix_a_p=A_p)
