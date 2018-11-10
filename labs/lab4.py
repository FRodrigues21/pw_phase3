#%%
import numpy as np
from numpy import linalg as LA
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
from sklearn.preprocessing import normalize
from skimage import color
from skimage import data, exposure
from numpy.random import shuffle

# Load mnist digits dataset using sklearn. Every digit image is represented as a 8x8 (64) RGB image.

from sklearn import datasets
mnist_digits = datasets.load_digits()
print("MNIST digits images shape: {}".format(mnist_digits.images[0].shape))
print("MNIST digits dataset shape: {}".format(mnist_digits.data.shape))
print("MNIST digits categories shape: {} - Categories C: {}".format(mnist_digits.target.shape, set(mnist_digits.target)))

indices = np.arange(len(mnist_digits.data))

# Shuffle the array - Modifies the array inplace 
shuffle(indices)

X = mnist_digits.data[indices]
y_target = mnist_digits.target[indices]

total_images = X.shape[0]

# Let's assume that 20% of the dataset is labeled
labeled_set_size = int(total_images*0.2)

indices_labeled = indices[:labeled_set_size]
indices_unlabeled = indices[labeled_set_size:]

print("Total images labeled: {} - Total images unlabeled: {}".format(len(indices_labeled), len(indices_unlabeled)))

# Convert labels to a one-hot-encoded vector
from keras.utils import to_categorical

# Keep groundtruth labels
Y_true = to_categorical(y_target)

Y = to_categorical(y_target)

# Remove labels of "unlabeled" data
Y[indices_unlabeled,:] = np.zeros(Y.shape[1])

def runAlg(images, y, y_true, feature, alpha, iterations, sigma, params=None):
    # Step 1 - Extract features for each image (HoG/CNN/HoC) in X
    if feature is "hog":
        pixels_per_cell=(2,2)
        orientations=8
        X = []
        for img in images:
            hist, hog_img = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, visualize=True, block_norm='L2-Hys')
            hist = np.squeeze(normalize(hist.reshape(1, -1), norm="l2"))
            X.append(hist)
        X = np.array(X)
        print("Shape of feature matrix: {}".format(X.shape))
    else:
        print("No feature selected")

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
        T1 = np.dot(T1,F)
        T2 = 1 * alpha * y_true
        F = T1 + T2
    F = np.argmax(F, axis=1)
    Y = to_categorical(F)
    return Y

# Run Algorithm and Get Results
feature = "hog"
alpha = 1
num_iterations = 300
sigma = 1
Y = runAlg(mnist_digits.images, Y, Y_true, feature=feature, alpha=alpha, iterations=num_iterations, sigma=sigma)

# Evaluation
from sklearn.metrics import classification_report
np.set_printoptions(threshold=np.nan)
# Get the predictions of the unlabeled documents
Y_pred = Y[indices_unlabeled, :]
#print(Y_pred)

# Get the corresponding groundtruth
y_gt = Y_true[indices_unlabeled, :]
#print(y_gt)
print("\n\nRESULTS FOR: F: {} a: {} i: {} s: {}\n".format(feature, alpha, num_iterations, sigma))
print(classification_report(y_gt, Y_pred))