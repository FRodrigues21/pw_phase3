# %%

# Hide all warnings
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pandas as pd
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
from sklearn.preprocessing import normalize
from skimage import color
from skimage import data, exposure
import random
from numpy.random import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import warnings
import os
warnings.filterwarnings('ignore')

print(os.getcwd(), "changing to:", os.getcwd()+"/../")

# Change this according to the path where you have the ws_toolkit
ws_toolkit_path = os.getcwd()+"/.."

os.chdir(ws_toolkit_path)
print(os.getcwd())
from ws_toolkit.utils import center_crop_image, k_neighbours

# Cached arrays
croppedImages = []

def cropImageList(images):
    global croppedImages
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


def features_hog(images, pixels_per_cell=(32, 32), orientations=8):
    gradientMatrix = []
    for img in images:
        # Convert to grayscale
        img_gray = rgb2gray(img)
        # Extract HoG features
        hist = hog(img_gray, orientations=orientations,
                   pixels_per_cell=pixels_per_cell)
        # Normalize features
        # We add 1 dimension to comply with scikit-learn API
        hist = np.squeeze(normalize(hist.reshape(1, -1), norm="l2"))
        gradientMatrix.append(hist)
    # Creating a feature matrix for all images
    gradientMatrix = np.array(gradientMatrix)
    return gradientMatrix


def runAlg(mlb, images, y, y_true, feature, alpha, iterations, sigma, params=None):
    # Step 1 - Extract features for each image (HoG/CNN/HoC) in X
    if feature is "hog":
        pixels_per_cell = (32, 32)
        orientations = 8
        X = features_hog(images, pixels_per_cell, orientations)
        # Step 2 - Initialize matrix F with Y
        F = y

        # Step 3 - Compute matrix W
        from sklearn.metrics.pairwise import euclidean_distances
        W = np.exp(-1*euclidean_distances(X, X)/(2*sigma**2))

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
            T2 = 1 * alpha * y_true
            F = T1 + T2
        # Select top k classes
        F = np.fliplr(np.argsort(F, axis=1))
        # Get the top 5 only classes
        F = F[:,:5]
        Y = mlb.transform(F)
        return Y
    else:
        print("No feature selected")
        return []


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
croppedImages = cropImageList(imageLinks)

# %%
def runAll(iterations):
    rand_seed = random.randint(1,100)
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
    labeled_set_size = int(total_images*0.2)

    indices_labeled = indices[:labeled_set_size]
    indices_unlabeled = indices[labeled_set_size:]

    print("Total tweets labeled: {} - Total tweets unlabeled: {}".format(
        len(indices_labeled), len(indices_unlabeled)))


    # Convert labels to a one-hot-encoded vector
    # Keep groundtruth labels
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # print(classes)
    mlb = MultiLabelBinarizer(classes=classes)
    Y_true = mlb.fit_transform(y_target)
    Y = mlb.transform(y_target)

    # Remove labels of "unlabeled" data
    Y[indices_unlabeled, :] = np.zeros(Y.shape[1])

    # Run Algorithm and Get Results
    feature = "hog"
    alpha = 1 # weight of the feature (in this case will be 1 = 100%)
    sigma = 1
    Y = runAlg(mlb, croppedImages, Y, Y_true, feature=feature,
            alpha=alpha, iterations=iterations, sigma=sigma)

    Y_pred = Y[indices_unlabeled, :]
    y_gt = Y_true[indices_unlabeled, :]
    return Y_pred, y_gt

from sklearn.metrics import precision_recall_fscore_support as score

results = np.zeros((49,4))
range_iterations = range(10,500,10)
j = 0

for i in range_iterations:
    Y_pred, y_gt = runAll(i)
    precision, recall, fscore, support = score(y_gt, Y_pred, average='macro')
    results[j] = [precision, recall, fscore, support]
    j = j+1

# %%
import matplotlib.pyplot as plt
plt.plot(range_iterations, results[:,0], 'r', range_iterations, results[:,1], 'b', range_iterations, results[:,2], 'g')

print(results)
