# %%
import numpy as np
import random as rnd
import nltk
import time
from numpy.random import shuffle
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MultiLabelBinarizer
from skimage import color
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from skimage import img_as_ubyte
from tokenizer import tokenizer
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
nltk.download('wordnet')


def center_crop_image(im, size=224):
    # Remove the alpha channel when present
    if len(im.shape) >= 3 and im.shape[2] == 4:
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


def bundle_crop(croppedImages, images, size=224):
    if len(croppedImages) <= 0:
        for imgName in images:
            croppedImg = []
            # Read image
            img = imread("./images/" + imgName)
            # Resize image
            croppedImg = center_crop_image(img, size)
            croppedImages.append(croppedImg)
    else:
        print("Using cached cropped images")
    return croppedImages


def k_neighbours(q, X, metric="euclidean", k=10):
    # Check pairwise_distances function docs:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
    dists = pairwise_distances(q, X, metric=metric)

    # Dists gets a shape 1 x NumDocs. Convert it to shape NumDocs (i.e. drop
    # the first dimension)
    dists = np.squeeze(dists)
    sorted_indexes = np.argsort(dists)

    return sorted_indexes[:k], dists[sorted_indexes[:k]]

# Histogram of Colors


def hoc(im, bins=(16, 16, 16), hist_range=(256, 256, 256)):
    im_r = im[:, :, 0]
    im_g = im[:, :, 1]
    im_b = im[:, :, 2]

    red_level = hist_range[0] / bins[0]
    green_level = hist_range[1] / bins[1]
    blue_level = hist_range[2] / bins[2]

    im_red_levels = np.floor(im_r / red_level)
    im_green_levels = np.floor(im_g / green_level)
    im_blue_levels = np.floor(im_b / blue_level)

    ind = im_blue_levels * bins[0] * bins[1] + \
        im_green_levels * bins[0] + im_red_levels

    hist_r, bins_r = np.histogram(ind.flatten(), bins[0] * bins[1] * bins[2])

    return hist_r, bins_r


def features_hoc(X_HOC, _croppedImages, _bins=(4, 4, 4), _hsv=True):
    # Histogram of colors results
    if X_HOC is None:
        X_HOC = []
        for img in _croppedImages:
            # Change image color space from RGB to HSV.
            # HSV color space was designed to more closely align with the way
            # human vision perceive color-making attributes
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

def extract_features_hoc(_queryImageId, _bins=(4,4,4)):
    img_q = imread("./query_images/" + _queryImageId)
    img_q_hsv = center_crop_image(img_q, size=224)
    img_q_hsv = color.rgb2hsv(img_q_hsv)
    img_int = img_as_ubyte(img_q_hsv)
    hist, bin_edges = hoc(img_int, bins=_bins)
    image_q_feat = np.squeeze(normalize(hist.reshape(1, -1), norm="l2"))
    return image_q_feat.reshape(1,-1)

def search_hoc(data_features, image_id, n_elements):
    query_features = extract_features_hoc(image_id)
    k_nearest_indexes, k_nearest_dists = k_neighbours(q=query_features, X=data_features, metric="cosine", k=n_elements)
    results = list(zip(k_nearest_indexes, k_nearest_dists))
    return results

# Histogram of Gradients

def features_hog(X_HOG, images, pixels_per_cell=(32, 32), orientations=8):
    if X_HOG is None:
        X_HOG = []
        for img in images:
            # Convert to grayscale
            img_gray = rgb2gray(img)
            # Extract HoG features
            hist = hog(img_gray, orientations=orientations,
                       pixels_per_cell=pixels_per_cell)
            # Normalize features
            # We add 1 dimension to comply with scikit-learn API
            hist = np.squeeze(normalize(hist.reshape(1, -1), norm="l2"))
            X_HOG.append(hist)
        # Creating a feature matrix for all images
        X_HOG = np.array(X_HOG)
    return X_HOG

def extract_features_hog(_queryImageId, _pixelsPerCell=(32,32), _orientations=8):
    img_q = imread("./query_images/" + _queryImageId)
    img_q = center_crop_image(img_q, size=224)
    img_q = rgb2gray(img_q)
    hist = hog(img_q, orientations=_orientations, pixels_per_cell=_pixelsPerCell)
    image_q_feat = np.squeeze(normalize(hist.reshape(1, -1), norm="l2"))
    return image_q_feat.reshape(1,-1)

def search_hog(data_features, image_id, n_elements):
    query_features = extract_features_hog(image_id)
    k_nearest_indexes, k_nearest_dists = k_neighbours(q=query_features, X=data_features, metric="cosine", k=n_elements)
    results = list(zip(k_nearest_indexes, k_nearest_dists))
    return results

# Bag of Words


def tokenizer_bow(sentence, tknzr, lemmatize=False):
    wnl = WordNetLemmatizer()
    tokens = []
    if lemmatize:
        tokens = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in [
            'a', 'n', 'v'] else wnl.lemmatize(i) for i, j in pos_tag(tknzr.tokenize(sentence))]
    else:
        tokens = tknzr.tokenize(sentence)
    return tokens

def bow_query(vectorizer, query):
    # Transform query in a BoW representation
    query_bow = vectorizer.transform([query])
    query_bow = normalize(query_bow, norm="l2")   
    return query_bow


def init_bow(texts, args, mdf):
    vectorizer = CountVectorizer(stop_words="english", min_df=mdf,
                                 binary=False, tokenizer=lambda text: tokenizer_bow(text, **args))
    texts_bow = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.vocabulary_
    #print("Vocabulary size: {}".format(len(vocabulary)))
    texts_bow = normalize(texts_bow, norm="l2")
    return vectorizer, texts_bow


def features_bow(X_BOW, _dataTexts, _lemmatize=False, _mdf=3, _metric="cosine", _k=10, _handles=False, _hashes=False, _case=False, _url=False):
    if X_BOW is None:
        tknzr = tokenizer.TweetTokenizer(
            preserve_handles=_handles, preserve_hashes=_hashes, preserve_case=_case, preserve_url=_url)
        X_BOW_VEC, X_BOW = init_bow(
            _dataTexts, {"tknzr": tknzr, "lemmatize": _lemmatize}, _mdf)
    return X_BOW, X_BOW_VEC

def search_bow(texts_bow, texts_vec, query_text, n_elements=10):
    query_bow = bow_query(texts_vec, query_text)
    k_nearest_indexes, k_nearest_dists = k_neighbours(q=query_bow, X=texts_bow, metric="cosine", k=n_elements)
    results = list(zip(k_nearest_indexes, k_nearest_dists))
    return results

# CNN - VGG 16


def process_images_keras(_images, _folder="./images/"):
    processedImgs = []
    for imgId in _images:
        # We are loading each image using Keras image and specifying the target
        # size.
        img = image.load_img(_folder + imgId, target_size=(224, 224))

        # Then the function img_to_array converts a PIL (library used by Keras)
        # to a numpy array
        x = image.img_to_array(img)

        # A one dimension is added to the the numpy array (224x224x3) becomes
        # (1x224x224x3)
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
    sorted_concepts = np.argsort(CNN_PREDICTIONS, axis=1)[:, ::-1][:, :k]
    data_tags = concepts
    mlb = MultiLabelBinarizer(classes=range(0, 1000))
    X_CNN = mlb.fit_transform(sorted_concepts)
    # print(tags_bow.shape)
    # model, concepts, mlb, tags_bow, predictions
    return model, concepts, mlb, X_CNN, CNN_PREDICTIONS 

def extract_features_cnn(_queryImageId, _model, _mlb, _k=10):
    print("Query", _queryImageId)
    query_list = process_images_keras([_queryImageId], "./query_images/")
    query_array_list = np.vstack(query_list)
    pred_query = _model.predict(query_array_list)
    query_sorted_concepts = np.argsort(pred_query, axis=1)[:,::-1][:,:_k]
    query_bow = _mlb.transform(query_sorted_concepts)
    query_tags = decode_predictions(pred_query, top=5)
    return query_bow, query_tags

def search_cnn(data_model, data_mlb, data_features, image_id, n_elements=10):
    query_features, query_tags = extract_features_cnn(image_id, data_model, data_mlb)
    k_nearest_indexes, k_nearest_dists = k_neighbours(q=query_features, X=data_features, metric="cosine", k=n_elements)
    results = list(zip(k_nearest_indexes,k_nearest_dists))
    return results


# Label Propagation


def label_propagation(mlb, images, y, y_true, features, weights, selection, topk, threshold, alpha, iterations, params=None, indices_unlabeled=[]):

    # Step 2 - Normalize Y and Initialize matrix F with Y
    Y_hidden = normalize(y, axis=1, norm="l1")
    F = Y_hidden

    # Step 3 - Compute matrix W (multi feature -mf true; or single feature -mf
    # false)
    W = []
    M = 0
    if(len(features) == 1):
        weights[0] = 1

    for i in range(len(features)):
        M += weights[i] * euclidean_distances(features[i], features[i])

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

def iteration_lb(data, targets, classes, iterations, p, features, weights, alpha, selection, topk, threshold):
    # Choose a random number between 1 and 100 to shuffle to prevent biased
    # results
    rand_seed = rnd.randint(1, 100)
    indices = np.arange(len(data))
    np.random.seed(rand_seed)
    shuffle(indices)

    X = data
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
    print("Iteration: {} - Total data labeled: {} - Total data unlabeled: {}".format(iterations,
                                                                                         len(indices_labeled), len(indices_unlabeled)))

    # Convert labels to a one-hot-encoded vector
    mlb = MultiLabelBinarizer(classes=classes)
    # print(y_target[:1])
    Y_true = mlb.fit_transform(y_target)
    Y = mlb.transform(y_target)
    # print(Y[:1])
    # Remove labels of "unlabeled" data
    Y[indices_unlabeled, :] = np.zeros(Y.shape[1])

    # Run Algorithm and Get Results
    Y = label_propagation(mlb, croppedImages, Y, Y_true, features=features, weights=weights, selection=selection, topk=topk, threshold=threshold,
                          alpha=alpha, iterations=iterations, indices_unlabeled=indices_unlabeled)

    Y_pred = Y[indices_unlabeled, :]
    y_gt = Y_true[indices_unlabeled, :]

    print("Ground Truth: {}".format(Y_true[indices_unlabeled[0], :]))
    print("Predicted:    {}".format(Y[indices_unlabeled[0], :]))

    return Y_pred, y_gt

# Rank Fusion

# CombSUM
# Input: Arrays of scores (4)
# Output: Array of document indexes sorted descending by final combSum score
def combSum(bow_results, hoc_results, hog_results, cnn_results):
    results = np.zeros(bow_results.shape)
    for i in range(len(results)):
        results[i] = bow_results[i] + hoc_results[i] + hog_results[i] + cnn_results[i]
    return np.flip(np.argsort(results))

# CombMNZ
# Input: Arrays of scores (4) AND Array of Top Doc Indexes
# Output: Array of document indexes sorted descending by final combMNZ score
def combMNZ(bow_results, hoc_results, hog_results, cnn_results, tops):
    results = np.zeros(bow_results.shape)
    for i in range(len(results)):
        r = 0
        for j in range(0, 4):
            if(np.isin(i, tops[j])):
                r+=1
        print("Index: {} _ R: {} ".format(i, r))
        results[i] = r * (bow_results[i] + hoc_results[i] + hog_results[i] + cnn_results[i])
    return np.flip(np.argsort(results))

tops = np.array([[1,2], [3,0], [4,3], [0,3]])

x = np.array([0.1, 0.5, 0.04, 0.1, 0.5])
y = np.array([0.4, 0.1, 0.1, 0.8, 0.1])
w = np.array([0, 0.1, 0.1, 0.3, 0.9])
z = np.array([0.5, 0.5, 0.1, 0.3, 0.5])

print(combSum(x,y,w,z))

print(combMNZ(x,y,w,z,tops))






