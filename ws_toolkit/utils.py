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



def k_neighbours(q, X, metric="euclidean", k=10):
    # Check pairwise_distances function docs: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
    dists = pairwise_distances(q, X, metric=metric)
    
    # Dists gets a shape 1 x NumDocs. Convert it to shape NumDocs (i.e. drop the first dimension)
    dists = np.squeeze(dists)
    sorted_indexes = np.argsort(dists)
    
    return sorted_indexes[:k], dists[sorted_indexes[:k]]

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