
# %%
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir("/Users/franciscorodrigues/Projects/PW/pw_phase3")
print("Current path: {}".format(os.getcwd()))
import time
import sys
import random
import pprint
import pandas as pd
import numpy as np
from dateutil.parser import parse
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
    "image-url").values, df.get("gt_class").values, df.get("created_at")])
# This are the text of the tweets
tweets = data[0]
# This are the links of the images of the tweets (ex: C0zsADasd213.jpg)
imageLinks = [i.replace('https://pbs.twimg.com/media/', '') for i in data[1]]
# This are the arrays of the data of each cropped image
targets = [list(map(int, c.replace(' ', '').split(","))) for c in data[2]]
# Get all the individual tweet dates and unique days dates
dates = pd.to_datetime(df["created_at"])
unique_dates = pd.to_datetime(df["created_at"]).dt.normalize().unique()
# Save cropped images in cache
croppedImages = bundle_crop(croppedImages, imageLinks, 224)

# Cache features
computeFeatures()

# PHASE 3
# TFIDF by days using the query
vectorizer = CountVectorizer(stop_words='english')
for text in tweets:
    tf = vectorizer.fit_transform([text])
    print(list(zip(vectorizer.get_feature_names(), np.ravel(tf.sum(axis=0)))))

# Sort by TFIDF value, and get top-k days