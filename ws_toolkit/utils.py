#%%
from skimage.transform import resize
from sklearn.metrics import pairwise_distances
from skimage import color
from keras.preprocessing import image
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import normalize
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
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

def center_crop_image(im, size=224):
    if len(im.shape) >= 3 and im.shape[2] == 4: # Remove the alpha channel when present
        im = im[:, :, 0:3]
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = resize(image=im, output_shape=(224, int(w * 224 / h)))
    else:
        im = resize(im, (int(h * 224 / w), 224))
    # Center crop to 224x224
    h, w, _ = im.shape
    im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]
    return im

def cropImageList(croppedImages, images):
    if len(croppedImages) <= 0:
        for imgName in images:
            croppedImg = []
            # Read image
            img = imread("./images/"+imgName)
            # Resize image
            croppedImg = center_crop_image(img, size=224)
            croppedImages.append(croppedImg)
    else:
        print("Using cached cropped images")
    return croppedImages

def k_neighbours(q, X, metric="euclidean", k=10):
    # Check pairwise_distances function docs: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
    dists = pairwise_distances(q, X, metric=metric)
    
    # Dists gets a shape 1 x NumDocs. Convert it to shape NumDocs (i.e. drop the first dimension)
    dists = np.squeeze(dists)
    sorted_indexes = np.argsort(dists)
    
    return sorted_indexes[:k], dists[sorted_indexes[:k]]

# Histogram of Colors
def hoc(im, bins=(16,16,16), hist_range=(256, 256, 256)):
    im_r = im[:,:,0]
    im_g = im[:,:,1]
    im_b = im[:,:,2]
    
    red_level = hist_range[0] / bins[0]
    green_level = hist_range[1] / bins[1]
    blue_level = hist_range[2] / bins[2]
    
    im_red_levels = np.floor(im_r / red_level)
    im_green_levels = np.floor(im_g / green_level)
    im_blue_levels = np.floor(im_b / blue_level)
    
    ind = im_blue_levels*bins[0]*bins[1]+ im_green_levels*bins[0] + im_red_levels
    
    hist_r, bins_r = np.histogram(ind.flatten(), bins[0]*bins[1]*bins[2])
    
    return hist_r, bins_r

def features_hoc(X_HOC, _croppedImages, _bins=(4,4,4), _hsv=True): 
    #Histogram of colors results
    if X_HOC is None:   
        X_HOC = []
        for img in _croppedImages:    
            # Change image color space from RGB to HSV. 
            # HSV color space was designed to more closely align with the way human vision perceive color-making attributes
            img_q = img
            if _hsv:
                img_q = color.rgb2hsv(img)    
            # convert image pixels to [0, 255] range, and to uint8 type
            img_q = img_as_ubyte(img_q)
            # Extract HoC features
            hist, bin_edges = hoc(img_q, bins=_bins)    
            # Normalize features
            # We add 1 dimension to comply with scikit-learn API
            hist = np.squeeze(normalize(hist.reshape(1, -1), norm="l2"))    
            X_HOC.append(hist)    
        # Creating a feature matrix for all images
        X_HOC = np.array(X_HOC)
    return X_HOC

# Histogram of Gradients
def features_hog(X_HOG, images, pixels_per_cell=(32, 32), orientations=8):
    if X_HOG is None:
        X_HOG = []
        for img in images:
            # Convert to grayscale
            img_gray = rgb2gray(img)   
            # Extract HoG features
            hist = hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell)   
            # Normalize features
            # We add 1 dimension to comply with scikit-learn API
            hist = np.squeeze(normalize(hist.reshape(1, -1), norm="l2"))  
            X_HOG.append(hist)
        # Creating a feature matrix for all images
        X_HOG = np.array(X_HOG)
    return X_HOG

# Bag of Words
def tokenizer_bow(sentence, tknzr, lemmatize=False):
    wnl = WordNetLemmatizer()
    tokens = []
    if lemmatize:
        tokens = [wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(tknzr.tokenize(sentence))]
    else:
        tokens = tknzr.tokenize(sentence)
    return tokens    

def init_bow(texts, args, mdf):
    vectorizer = CountVectorizer(stop_words="english", min_df=mdf, binary=False, tokenizer=lambda text: tokenizer_bow(text, **args))
    texts_bow = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.vocabulary_
    #print("Vocabulary size: {}".format(len(vocabulary)))
    texts_bow = normalize(texts_bow, norm="l2")    
    return vectorizer, texts_bow

def features_bow(X_BOW, _dataTexts, _lemmatize=False, _mdf=3, _metric="cosine", _k=10, _handles=False, _hashes=False, _case=False, _url=False):
    if X_BOW is None:
        tknzr = tokenizer.TweetTokenizer(preserve_handles=_handles, preserve_hashes=_hashes, preserve_case=_case, preserve_url=_url)
        vectorizer, X_BOW = init_bow(_dataTexts, {"tknzr": tknzr, "lemmatize": _lemmatize}, _mdf)
    return X_BOW

# CNN - VGG 16
def process_images_keras(_images, _folder="./images/"):
    processedImgs = []
    for imgId in _images:
        # We are loading each image using Keras image and specifying the target size.
        img = image.load_img(_folder + imgId, target_size=(224, 224))

        # Then the function img_to_array converts a PIL (library used by Keras) to a numpy array
        x = image.img_to_array(img)

        # A one dimension is added to the the numpy array (224x224x3) becomes (1x224x224x3)
        x = np.expand_dims(x, axis=0)

        # Apply Keras pre-processing steps specific to VGG16 model
        x = preprocess_input(x)
        processedImgs.append(x)
    return processedImgs

def features_cnn(X_CNN, CNN_PREDICTIONS, images_path, _dataImages):
    model = VGG16(weights='imagenet', include_top=True)
    start = time.time()
    if len(CNN_PREDICTIONS) <= 0:
        start_i = time.time()
        img_list = process_images_keras(_dataImages, images_path)
        end_i = time.time()
        print("Processed Images finished: {}".format(end_i - start_i))
        #model = ResNet50(weights='imagenet')
        # Convert from list to ndarray
        img_array_list = np.vstack(img_list)
        # Feed all images to the model
        print("No Cached Predictions")
        CNN_PREDICTIONS = model.predict(img_array_list)
        print("Finished dataset predictions")
    else:
        print("Using Cached Predictions")
    end = time.time()
    print("Model Predictions finished: {}".format(end - start))
    #print("Resulting shape of the network output: {}".format(preds.shape))
    concepts = decode_predictions(CNN_PREDICTIONS, top=5)
    # Experiment with this parameter
    k = 5
    # Get the top K most probable concepts per image
    sorted_concepts =  np.argsort(CNN_PREDICTIONS, axis=1)[:,::-1][:,:k]
    data_tags = concepts
    mlb = MultiLabelBinarizer(classes=range(0,1000))
    X_CNN = mlb.fit_transform(sorted_concepts)
    #print(tags_bow.shape)
    return X_CNN, CNN_PREDICTIONS
