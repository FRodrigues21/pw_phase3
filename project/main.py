
# %%
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir("/Users/franciscorodrigues/Projects/PW/pw_phase3")
print("Current path: {}".format(os.getcwd()))
import time
import sys
import random
import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy.random import shuffle
from tokenizer import tokenizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MultiLabelBinarizer
from skimage import color, data, exposure
from skimage import img_as_ubyte
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import ws_toolkit.utils

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
croppedImages = bundle_crop(croppedImages, imageLinks, 224)

# Cache features
computeFeatures()

# %%

results = np.zeros((10, 4))
range_iterations = range(10, 500, 50)
j = 0

# Variable params
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
p = 0.5
# Possible X_HOG, X_HOG, X_BOW, X_CNN -> If mf=False use only one feature
# inside the array
features = [X_BOW, X_CNN]
# Sum must be one - and must always be filled even if mf=False
weights = [0.6, 0.4]
alpha = 0.2
selection = False  # False - Top K | True - Threshold
topk = 3
threshold = 0.07

for i in range_iterations:
    Y_pred, y_gt = iteration_lb(tweets, targets, classes, i, p, features, weights, alpha,
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
