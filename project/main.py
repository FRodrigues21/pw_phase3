#%%
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tokenizer import tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pandas as pd
from sklearn.preprocessing import normalize
from skimage import color
from skimage import data, exposure
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
import sys
import random
from numpy.random import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from skimage import img_as_ubyte
import warnings
import os
import time
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image

warnings.filterwarnings('ignore')

from ws_toolkit.utils import *

# Cached arrays
croppedImages = []
X_HOC = None
X_HOG = None
X_BOW = None
X_CNN = None
CNN_PREDICTIONS = []


def computeFeatures():
    global X_BOW, X_HOG, X_CNN, X_HOC, CNN_PREDICTIONS
    pixels_per_cell = (32, 32)
    orientations = 8

    bins = (4, 4, 4)
    hsv = True

    _lemmatize = False
    _mdf = 3
    _metric = "cosine"
    _k = 10
    _handles = False
    _hashes = False
    _case = False
    _url = False

    X_HOG = features_hog(X_HOG, croppedImages, pixels_per_cell, orientations)
    X_HOC = features_hoc(X_HOC, croppedImages, bins, hsv)
    X_BOW = features_bow(X_BOW, tweets, _lemmatize, _mdf,
                         _metric, _k, _handles, _hashes, _case, _url)
    X_CNN, CNN_PREDICTIONS = features_cnn(
        X_CNN, CNN_PREDICTIONS, "./images/", imageLinks)

#%%

# Read dataset .csv
df = pd.read_csv("./visualstories_edfest_2016_twitter_xmedia.csv",
                 sep=';', encoding="utf-8")

data = np.array([df.get("text").values, df.get(
    "image-url").values, df.get("gt_class").values])
# This are the text of the tweets
tweets = data[0]
# This are the links of the images of the tweets (ex: C0zsADasd213.jpg)
imageLinks = [i.replace('https://pbs.twimg.com/media/', '') for i in data[1]]
# This are the arrays of the data of each cropped image
targets = [list(map(int, c.replace(' ', '').split(","))) for c in data[2]]
# Save cropped images in cache
croppedImages = cropImageList(croppedImages, imageLinks)

# Cache features
computeFeatures()


def runAlg(mlb, images, y, y_true, features, weights, selection, topk, threshold, alpha, iterations, params=None, indices_unlabeled=[]):

    # Step 2 - Normalize Y and Initialize matrix F with Y
    Y_hidden = normalize(y, axis=1, norm="l1")
    F = Y_hidden
    # print(F[indices_labeled[0],:])

    # Step 3 - Compute matrix W (multi feature -mf true; or single feature -mf
    # false)
    W = []
    M = 0
    print("Features: {}".format(len(features)))
    if(len(features) == 1):
        weights[0] = 1
    print(weights[0])
    for i in range(len(features)):
        M += weights[i] * euclidean_distances(features[i], features[i])
    #M = weights[0]*euclidean_distances(feature[0], feature[0]) + weights[1]*euclidean_distances(feature[1], feature[1])

    sigma = np.std(M)
    W = np.exp(-1 * M / (2 * sigma**2))

    # Step 4 - Normalize W
    D = np.zeros(W.shape)
    np.fill_diagonal(D, W.sum(axis=0))

    D12 = np.zeros(D.shape)
    from numpy.linalg import inv
    D12 = inv(np.sqrt(D))

    S = np.dot(D12, W)
    S = np.dot(S, D12)

    # Step 5 - Perform the F update step num_iterations steps
    for i in range(1, iterations):
        T1 = alpha * S

        T1 = np.dot(T1, F)
        T2 = (1 - alpha) * Y_hidden
        F = T1 + T2
        # Normalizar para F (verficar segmentos)
        F = normalize(F, axis=1, norm="l1")

    print("Indice unlabeled: {}\nNormalized F: {}".format(
        indices_unlabeled[0], F[indices_unlabeled[0], :]))
    # Select top k classes
    if selection is True:
        F = np.fliplr(np.argsort(F, axis=1))
        F = F[:, :topk]
        Y = mlb.transform(F)
    else:
        T = []
        for row in F:
            T.append([i for i, v in enumerate(row) if v >= threshold])
        Y = mlb.transform(T)
    return Y


def runAll(iterations, p, features, weights, alpha, selection, topk, threshold):
    # Choose a random number between 1 and 100 to shuffle to prevent biased
    # results
    rand_seed = random.randint(1, 100)
    indices = np.arange(len(tweets))
    np.random.seed(rand_seed)
    shuffle(indices)

    X = tweets
    np.random.seed(rand_seed)
    shuffle(X)

    y_target = targets
    np.random.seed(rand_seed)
    shuffle(y_target)

    total_images = X.shape[0]

    # Let's assume that 20% of the dataset is labeled
    labeled_set_size = int(total_images * p)

    indices_labeled = indices[:labeled_set_size]
    indices_unlabeled = indices[labeled_set_size:]

    print(" ")
    print("Iteration: {} - Total tweets labeled: {} - Total tweets unlabeled: {}".format(iterations,
                                                                                         len(indices_labeled), len(indices_unlabeled)))

    # Convert labels to a one-hot-encoded vector
    # Keep groundtruth labels
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # print(classes)
    mlb = MultiLabelBinarizer(classes=classes)
    # print(y_target[:1])
    Y_true = mlb.fit_transform(y_target)
    Y = mlb.transform(y_target)
    # print(Y[:1])
    # Remove labels of "unlabeled" data
    Y[indices_unlabeled, :] = np.zeros(Y.shape[1])

    # Run Algorithm and Get Results
    Y = runAlg(mlb, croppedImages, Y, Y_true, features=features, weights=weights, selection=selection, topk=topk, threshold=threshold,
               alpha=alpha, iterations=iterations, indices_unlabeled=indices_unlabeled)

    Y_pred = Y[indices_unlabeled, :]
    y_gt = Y_true[indices_unlabeled, :]

    print("Ground Truth: {}".format(Y_true[indices_unlabeled[0], :]))
    print("Predicted:    {}".format(Y[indices_unlabeled[0], :]))

    return Y_pred, y_gt

results = np.zeros((10, 4))
range_iterations = range(10, 500, 50)
j = 0

# Variable params
p = 0.7
# Possible X_HOG, X_HOG, X_BOW, X_CNN -> If mf=False use only one feature
# inside the array
features = [X_BOW, X_CNN]
# Sum must be one - and must always be filled even if mf=False
weights = [0.5, 0.5]
alpha = 0.8
selection = False  # False - Top K | True - Threshold
topk = 3
threshold = 0.15

for i in range_iterations:
    Y_pred, y_gt = runAll(i, p, features, weights, alpha,
                          selection, topk, threshold)
    precision, recall, fscore, support = score(y_gt, Y_pred, average='macro')
    results[j] = [precision, recall, fscore, support]
    j = j + 1

from sklearn.metrics import classification_report
np.set_printoptions(threshold=np.nan)

print("\nResults List:    \n{}".format(results))
print("\nResults Report:  \n{}".format(classification_report(y_gt, Y_pred)))

print("\nResults Graph:   \n")

colors = ['r', 'b', 'g']
labels = ['precision', 'recall', 'f-score']

for i in range(3):
    plt.plot(range_iterations, results[:, i], colors[i], label=labels[i])
plt.xlabel("Iterations")
plt.ylabel("Values")
plt.legend(loc='best')
plt.show()
#plt.plot(range_iterations, results[:,0], 'r', range_iterations, results[:,1], 'b', range_iterations, results[:,2], 'g')
