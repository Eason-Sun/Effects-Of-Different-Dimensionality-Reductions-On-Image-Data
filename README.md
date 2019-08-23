# Effects-Of-Different-Dimensionality-Reductions-On-Image-Data

## Dataset:
Handwritten digits of 0, 1, 2, 3, and 4 (5 classes). This dataset contains 2066 samples with 784 features corresponding to a 28 x 28 gray-scale (0-255) image of the digit, arranged in column-wise.

## Linear Dimensionality Reduction Techniques:

### (a) Principal Component Analysis (PCA)
Implement PCA to reduce Dimensionality of image files. Plot a 2 dimensional representation of the data points based on the 1st, 2nd, 5th and 6th principal components. Naive Bayes classifier is used to classify 8 sets of dimensionality reduced data (using the 1, 2, 4, 10, 30, 60, 200, 500, and all 784 PCA components). Classification error is plotted for the 8 sets against the retained variance of each case.

2-dimensional representation with 1st and 2nd principle components:
![ECE_657A_Assignment1](https://user-images.githubusercontent.com/29167705/63562457-ae4c1c80-c52b-11e9-9e4b-f1fceebf5516.jpg)

2-dimensional representation with 5th and 6th principle components.
![ECE_657A_Assignment1](https://user-images.githubusercontent.com/29167705/63562479-c459dd00-c52b-11e9-892b-8600947e3d84.jpg)

As we can see from two graphs above. We observe that data tend to have lower variance as the order of principle components decrease. For example, the approximate range for PC1 is [-20, 10], PC2, PC5 and PC6 have the approximate ranges of [-15, 10], [-12, 11] and [-8, 10] respectively. We can clearly see that the approximate range of variation shrinks as PC order declines.

The following figure illustrates how the number of principle components taken from PCA can affect the accuracy of the Naive Bayes Classification. The error rate drops drastically and reaches it minimum (9%) when 30 PC components are taken, then it increase logarithmically. Lastly, it tends to decline steadily. This pattern can be interpreted as following: When there are very few principle components, data are overly summarized, and details might get lost. Thus, the Naive Bayes Classification performs poorly at the beginning. When there are too many principle components, the performance also get degraded because the higher order PC does not contain useful information and can be regarded as noise.

![ECE_657A_Assignment1](https://user-images.githubusercontent.com/29167705/63562839-0e8f8e00-c52d-11e9-8088-e49ebe8cbb5c.jpg)

### (a) Linear Discriminant Analysis (LDA)

As shown in Figure below, Linear Discriminant Analysis not only reduces the dimension but also makes the data with different classes more separable from each other. As the above graph shown, the only overlapped region is between 2 and 3, which is probably due to the fact that 2 and 3 look alike the most among all the given digits. On the other hand, PCA might not perform very well in terms of
the data separation because it seeks for the maximum variance, which does not necessarily reflect the separability. In other word, different classes could be mainly affected by some slow changing features.

![ECE_657A_Assignment1](https://user-images.githubusercontent.com/29167705/63562962-6c23da80-c52d-11e9-8e6c-ce92d7914555.jpg)


## Nonlinear Dimensionality Reduction Techniques:

### (a) Locally Linear Embedding (LLE)
Apply LLE to the images of digit 3 only. The original images are visualized by plotting the images corresponding to those instances on 2-D representations of the data based on the first and second components of LLE.
The number of nearest neighbours is to be 5, and the projected low dimension is set to be 4.

![ECE_657A_Assignment1](https://user-images.githubusercontent.com/29167705/63563211-477c3280-c52e-11e9-8402-2f8a87003743.jpg)

The Locally Linear Embedding groups the similar variation together. If we look at the left part of the plot, we can observe that these digits are rotated clockwise by some degree. While in the lower right corner, those digits are rotated counterclockwise. In addition, digits on the top look far and thin whereas those on the bottom look near and wide.

### (b) ISOMAP
Instead of computing the local weight matrix from the k-nearest neighbors, ISOMAP calculates the pairwise geodesic distance globally. It accurately analyze the manifold pattern so that digits in their own clusters look more similar among all variations.

![ECE_657A_Assignment1](https://user-images.githubusercontent.com/29167705/63563309-a3df5200-c52e-11e9-9e48-fcd79f3918d5.jpg)

## Overall Evaluation:
The Naive Bayes classifier to classify the dataset based on the projected 4-dimension representations of the PCA, LDA, LLE and ISOMAP.

![Capture](https://user-images.githubusercontent.com/29167705/63563439-1819f580-c52f-11e9-90a8-b899ae64dacd.JPG)

## Conclusion: 
From the above chart, LDA is definitely the best performed dimensional reduction approach in terms of classification accuracy provided by Gaussian Naive Bayess classifier. The nonlinear approaches such as LLE and ISOMAP also have decent performance in classification.
Moreover, they provide additional manifold information within the same class, like poses and orientations. PCA is aimed to find the maximum variations that might contain useful information , but its not necessary to be able to separate the data very well.

