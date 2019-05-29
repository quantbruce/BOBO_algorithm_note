import numpy as np
from math import sqrt
from collections import Counter


def kNN_classify(k, X_train, y_train, x):

    assert 1<= k <= X_train.shape[0], "k must be vaild"
    assert X_train.shape[0] == y_train.shape[0] , \
             "the size of X_train must equal to the size of y_train"
    assert x.shape[1] == X_train.shape[1], \
            "the feature number of x must be equal to X_train"

    distances = [sqrt(np.sum((x - x_train)**2)) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]
