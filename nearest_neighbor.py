import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.vq import whiten
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.utils.validation import DataConversionWarning

x = np.array([[0, 0], [0, 1], [5, 5]])
y = np.array([1,1,-1])


def compare(x_1, x_2):
    return np.all(np.isclose(x_1, x_2))


def run_comparison():
    vector = x
    bias = vector + 2
    uniform = vector * 2
    scaled = np.dot(np.diag(np.random.rand(1, len(vector))[0]), vector)
    linear_transform = np.dot(np.arange(len(vector) * len(vector)).reshape(len(vector), len(vector)), vector)

    print('\t\t       bias, unif, scale, linTran')
    centered = vector - np.mean(vector)
    print('Centering:  | {0}, {1}, {2}, {3}'.format(compare(bias - np.mean(bias), centered),
                                                 compare(uniform - np.mean(uniform), centered),
                                                 compare(scaled - np.mean(scaled), centered),
                                                 compare(linear_transform - np.mean(linear_transform), centered)))

    try:
        normalized = normalize(vector)
        print('Normalizing:| {0}, {1}, {2}, {3}'.format(compare(normalize(bias), normalized),
                                                     compare(normalize(uniform), normalized),
                                                     compare(normalize(scaled), normalized),
                                                     compare(normalize(linear_transform), normalized)))
    except DataConversionWarning:
        pass

    whitened = whiten(vector)
    print('Whitening:  | {0}, {1}, {2}, {3}'.format(compare(whiten(bias), whitened),
                                                   compare(whiten(uniform), whitened),
                                                   compare(whiten(scaled), whitened),
                                                   compare(whiten(linear_transform), whitened)))


# xx = normalize(x_data)
# xx = whiten(xx)
# print(xx)


def getboundry(x_data, y_data, doplot=False):
    # pca = PCA(n_components=1, whiten=True)
    # xx = pca.fit_transform(np.reshape(x_data, (len(x_data), 1)), y_data)

    step = .01
    temp = []
    for item in x_data:
        temp.append([item, 1])
    x = np.array(temp)

    clf = neighbors.KNeighborsClassifier(n_neighbors=1000, weights='uniform')
    clf.fit(x, y_data)

    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    # Plot the decision boundary.
    x_min, x_max = min(x[:, 0]), max(x[:, 0])
    y_min, y_max = -0.1, 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    if doplot:
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot the training points
        plt.scatter(x[:, 0], x[:, 1], c=y_data, cmap='RdBu')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("NN classification, whitened (k = %i, weights = '%s')"
                  % (2, 'distance'))

    for zz in range(len(Z[0])):
        if Z[0][zz] > 0:
            boundry = xx[0][zz]
            return boundry

    return 0


def plot():
    plt.show()