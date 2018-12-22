
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
# %% 
import ws_toolkit.utils
# %%
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

    _lemmatize = True
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

#%%
# Cache features
computeFeatures()

#%%
# RANK FUSION FROM QUERIES
import json

with open("./edfest_2016_stories.json") as f:
    data = json.load(f)

stories = data["stories"][:5]
for story in stories:
    print("\n=== STORY: {}===\n".format(story["story_title"]))
    segments = story["segments"]
    for seg in segments:
        seg_id = seg["segment_id"]
        seg_text = seg["text"]
        seg_img = seg["image"]+".jpg"
        print("SEGMENT #{} - Text: `{}` - Image: {}\n".format(seg_id, seg_text, seg_img))
        elements = tweets.shape[0]
        top = 50
        results_bow = search_bow(X_BOW, X_BOW_VEC, seg_text, elements)
        results_hoc = search_hoc(X_HOC, seg_img, elements)
        results_hog = search_hog(X_HOG, seg_img, elements)
        results_cnn = search_cnn(CNN_MODEL, CNN_MLB, CNN_TAGS, seg_img, elements)
        search_results = [results_bow, results_hoc, results_hog, results_cnn]
        print(rrf(search_results, elements, 60))
        
