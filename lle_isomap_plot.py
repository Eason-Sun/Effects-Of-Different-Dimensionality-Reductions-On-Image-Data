import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold

df = pd.read_csv('DataB.csv', index_col=0)
class_matrix = df['gnd'].values

# Slice the data matrix so that only samples in class '3' are taken.
digit_3_df = df.loc[df['gnd'] == 3].copy()
digit_3_df.drop(columns=['gnd'], inplace=True)
digit_3_matrix = digit_3_df.values

df.drop(columns=['gnd'], inplace=True)
data_matrix = df.astype(float).values
n_neighbors = 5  # Set the number of nearest neighbour to 5.

# Plot the image based on the first and second components of LLE or ISOMAP.
def plot(X, title=None, min_dist=4e-3):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)  # Min-Max Normalization.
    plt.figure()
    ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(X.shape[0]):
        # Discard those samples that appear too near in the figure.
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < min_dist:
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        # Map each image to their corresponding coordinates provided by the first 2 components of the projected matrix.
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digit_3_matrix[i].reshape((28, 28)), cmap=plt.cm.gray_r), X[i],
                                            frameon=False)
        ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# Apply LLE to the dataset in class '3'.
lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components=4, method='standard')
lle.fit(digit_3_matrix)
X_lle = lle.transform(digit_3_matrix)
plot(X_lle[:, 0:2], "LLE Projection", 3e-3)

# Apply ISOMAP to the dataset in class '3'.
iso = manifold.Isomap(n_neighbors, n_components=4)
iso.fit(digit_3_matrix)
X_iso = iso.transform(digit_3_matrix)
plot(X_iso[:, 0:2], "ISOMAP Projection")

plt.show()
