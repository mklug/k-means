# K-means

This repository contains an implementation of the k-means clustering algorithm (see ``KMeans.py``).  We largely follow the [sklearn API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and we allow for the initialization of the points either manually, through a random selection of points in the training data, or through [k-means++ initialization](https://en.wikipedia.org/wiki/K-means%2B%2B).  Optimization is done following Lloyd's algorithm.

The notebook ``image_compression.ipynb`` contains an application of k-means clustering to image quantization.  In particular, we take the jpeg image in ``moai.jpg`` and convert it to an ordered pointcloud of 3-dimensional points (the RGB coordinates).  This is done using the [opencv library](https://pypi.org/project/opencv-python/) and as such, that is a dependency (along with the usual numpy, pandas, matplotlib).  From this set of points, we find the k-means with $k=10$ and then we show the image where each pixel is replaced by the mean that is closest.  This is in a way the best approximation of the original image that uses only 10 colors.  

Prediction time is currently slow but could be sped up considerably by using a [kd tree](https://en.wikipedia.org/wiki/K-d_tree) to store the means and to query for nearest neighbors.  
