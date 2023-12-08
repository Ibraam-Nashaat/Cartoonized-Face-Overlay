import numpy as np
from skimage import feature, color, transform
from utils import *

# Load your image
# image = skimage.io.imread('path_to_your_image.jpg')

# Convert the image to grayscale
gray_image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Compute the integral image
# transform.integral.integral_image(gray_image)
integral_image = [[1, 2, 3, 8], [4, 5, 6, 7], [9, 10, 11, 12]]

# Define the region of interest (ROI)
r, c, width, height = 0, 0, 2, 2  # Example values

# Get the coordinates of Haar-like features
feature_coord, _ = feature.haar_like_feature_coord(width, height, 'type-2-x')

img = np.ones((3, 3), dtype=np.uint8)
img_ii = transform.integral_image(img)
print(img_ii)

utils = Utils()
integral_image = utils.get_integral_image(
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
integral_image = np.array(integral_image, dtype=int)

feature_coord, feature_type = zip(
    *[feature.haar_like_feature_coord(3, 3, feat_t)
      for feat_t in ('type-2-y', 'type-2-x', 'type-3-y', 'type-3-x', 'type-4')])
print(feature_coord)

feature_coord = np.concatenate([x[::2] for x in feature_coord])
feature_type = np.concatenate([x[::2] for x in feature_type])


# Compute the Haar-like features
features = feature.haar_like_feature(integral_image, 1, 1, 4, 4,
                                     feature_type=feature_type,
                                     feature_coord=feature_coord)
print(features)
# 'features' now contains the Haar-like features of the specified ROI
