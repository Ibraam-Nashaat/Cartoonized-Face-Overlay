import skimage.io as io
from skimage.color import rgb2gray
import numpy as np
from haar_like_features import *
from errors import *
import pytest


def test_haar_like_feautures_4x4_matched():
    image = np.array([[1, 5, 7, 2], [9, 11, 8, 3],
                     [3, 2, 5, 6], [10, 9, 8, 7]])

    haar_like_feautures = HaarLikeFeatures()
    features = np.array(haar_like_feautures.extract_features(
        image, 0, 0, 4, 4), dtype=int)
    expected_features = np.array([-8, -6, -1, -1, 6, 9, 3, -3, -7, -7, -3, -1, 2, 7, -6, -4, -2, 5,
                                  -2, 3, 5, 1, -3, -1, 1, 1, 1, 6, 3, -2, -5, -4, 4, 5, 16, 18, 11,
                                  4, 3, 0, 6, 6, 6, 3, 9, 8, -2, -5, 0, -3, 6, 6, 0, -4, -2, 8], dtype=int)

    assert np.array_equal(features,
                          expected_features), "Haar Features do not match!"


def test_haar_like_feautures_3x3_matched_t1():
    image = np.array([[1, 5, 7], [9, 11, 8], [3, 2, 5]])

    haar_like_feautures = HaarLikeFeatures()
    features = np.array(haar_like_feautures.extract_features(
        image, 0, 0, 3, 3), dtype=int)

    expected_features = np.array(
        [-8, -6, -1, 6, 9, 3, -4, -2, -2, 3, 1, -3, -5, -4, 4, 3, 6, 6, -2, -5, -3, 6], dtype=int)

    assert np.array_equal(features,
                          expected_features), "Haar Features do not match!"


def test_haar_like_feautures_3x3_matched_t2():
    image = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    haar_like_feautures = HaarLikeFeatures()
    features = np.array(haar_like_feautures.extract_features(
        image, 0, 0, 3, 3), dtype=int)

    expected_features = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int)

    assert np.array_equal(features,
                          expected_features), "Haar Features do not match!"


def test_haar_like_feautures_on_real_image():
    image = io.imread("Images/hermoine.jpg")
    image = (rgb2gray(image)*255).astype('uint8')
    haar_like_feautures = HaarLikeFeatures()
    features = np.array(haar_like_feautures.extract_features(
        image, 0, 0, 10, 15), dtype=int)


def test_haar_like_feautures_empty_image():
    image = np.array([])
    haar_like_feautures = HaarLikeFeatures()
    errors = Errors()

    with pytest.raises(ValueError, match=errors.get_empty_image_message()):
        features = np.array(haar_like_feautures.extract_features(
            image, 0, 0, 3, 3), dtype=int)


def test_haar_like_feautures_negative_window_dimension():
    image = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    haar_like_feautures = HaarLikeFeatures()
    errors = Errors()

    with pytest.raises(ValueError, match=errors.get_negative_dimensions_message()):
        features = np.array(haar_like_feautures.extract_features(
            image, 0, 0, -1, 3), dtype=int)

    with pytest.raises(ValueError, match=errors.get_negative_dimensions_message()):
        features = np.array(haar_like_feautures.extract_features(
            image, 0, 0, 3, -1), dtype=int)

    with pytest.raises(ValueError, match=errors.get_negative_dimensions_message()):
        features = np.array(haar_like_feautures.extract_features(
            image, 0, 0, -1, -2), dtype=int)


def test_haar_like_feautures_window_outside_image_bounds():
    image = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    haar_like_feautures = HaarLikeFeatures()
    errors = Errors()

    with pytest.raises(ValueError, match=errors.get_window_out_of_bounds_message()):
        features = np.array(haar_like_feautures.extract_features(
            image, 0, 0, 4, 3), dtype=int)

    with pytest.raises(ValueError, match=errors.get_window_out_of_bounds_message()):
        features = np.array(haar_like_feautures.extract_features(
            image, 0, 0, 3, 4), dtype=int)

    with pytest.raises(ValueError, match=errors.get_window_out_of_bounds_message()):
        features = np.array(haar_like_feautures.extract_features(
            image, 0, 0, 4, 4), dtype=int)
