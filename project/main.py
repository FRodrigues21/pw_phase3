
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
X_BOW_VEC = None
CNN = None
CNN_MODEL = None
CNN_CONCEPTS = None
CNN_MLB = None
CNN_TAGS = None
CNN_PREDICTIONS = []


def computeFeatures():
    global X_BOW, X_BOW_VEC, X_HOG, X_HOC, CNN_MODEL, CNN_CONCEPTS, CNN_MLB, CNN_TAGS, CNN_PREDICTIONS
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
    X_BOW, X_BOW_VEC = features_bow(X_BOW, tweets, _lemmatize, _mdf,
                                    _metric, _k, _handles, _hashes, _case, _url)
    CNN_MODEL, CNN_CONCEPTS, CNN_MLB, CNN_TAGS, CNN_PREDICTIONS = features_cnn(
        CNN, CNN_PREDICTIONS, "./images/", imageLinks)

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

# Rank Fusion
"""
1. For each query perform search in
    2. Each search space (BoW, HoC, HoG, CNN):
        - Save each search space scores in a list
    3. Apply each rank fusion method
        a) Save final score in array
    4. Sort documents - array a) - by rank fusion score
"""
# %%
elements = 2000
results_bow = search_bow(X_BOW, X_BOW_VEC, "The Event Edinburgh Castle", elements)
results_hoc = search_hoc(X_HOC, "760513155030220800.jpg", elements)
results_hog = search_hog(X_HOG, "760513155030220800.jpg", elements)
results_cnn = search_cnn(CNN_MODEL, CNN_MLB, CNN_TAGS,
                         "760513155030220800.jpg", elements)

results = [results_bow, results_cnn, results_hoc, results_hog]

# COMBSUM
print("\n--- COMBSUM RESULTS ---\n")
combsum = []
for doc_id in range(0, elements):
    sum = 0
    for s in range(0, len(results)):
        try:
            doc, score, position = next((doc, score, position) for (
                doc, score, position) in results[s] if doc == doc_id)
            sum += score
        # When document not found (only for tops I guess)
        except StopIteration:
            sum += 0
    combsum.append((doc_id, imageLinks[doc_id], sum))
print(sorted(combsum, key=lambda s: s[2], reverse=True))

# COMBMNZ
print("\n--- COMBMNZ RESULTS ---\n")
top = 5
combmnz = []
for doc_id in range(0, 2000):
    sum = 0
    cnt = 0
    for s in range(0, len(results)):
        try:
            doc, score, position = next((doc, score, position) for (
                doc, score, position) in results[s] if doc == doc_id)
            sum += score
            if position <= top:
                cnt += 1
        # When document not found (only for tops I guess)
        except StopIteration:
            sum += 0
    combmnz.append((doc_id, imageLinks[doc_id], cnt * sum))
print(sorted(combmnz, key=lambda s: s[2], reverse=True))

# POSITION BASED
# BORDA-FUSE
print("\n--- BORDAFUSE RESULTS ---\n")
bordafuse = []
for doc_id in range(0, elements):
    sum = 0
    for s in range(0, len(results)):
        try:
            doc, score, position = next((doc, score, position) for (
                doc, score, position) in results[s] if doc == doc_id)
            sum += elements - position
        # When document not found (only for tops I guess)
        except StopIteration:
            sum += elements + 1
    bordafuse.append((doc_id, imageLinks[doc_id], sum))
print(sorted(bordafuse, key=lambda s: s[2], reverse=True))


# Parse query news topics/segments
