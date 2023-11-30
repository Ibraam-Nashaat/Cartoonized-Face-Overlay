import numpy as np
from utils import *
from enum import Enum
from math import ceil


class HaarFeautureTypes(Enum):
    TWO_HORIZONTAL = 0
    TWO_VERTICAL = 1
    THREE_HORIZONTAL = 2
    THREE_VERTICAL = 3
    FOUR_DIAGONAL = 4


class HaarLikefeatures:

    def __init__(self):
        self.haar_window = [
            (2, 1),
            (1, 2),
            (3, 1),
            (1, 3),
            (2, 2)
        ]

        self.utils = Utils()

    def get_sum_in_rectangle(self, integral_image, start_row, start_col, w, h):
        """
        Gets the sum of pixels in rectangle

        Parameters: 
        integral_image: np.array
        start_row: int
                   start row of rectangle relative to the original image
        start_col: int 
                   start column of rectangle relative to the origial image
        w: int
           width of the rectangle
        h: int
           height of the rectangle

        Returns:
        sum: int
             sum of the pixels inside the original image rectangle
        """
        start_row, start_col, w, h = \
            int(start_row), int(start_col), int(w), int(h)
        sum = integral_image[start_row+h, start_col+w] + \
            integral_image[start_row, start_col] - \
            integral_image[start_row+h, start_col] - \
            integral_image[start_row, start_col+w]
        return int(sum)

    def get_feauture_value(self, integral_image, feauture_type, start_row, start_col, w, h):
        """
        Gets the value of a cerain haar feauture type

        Parameters:
        integral_image: np.array
        feauture_type: HaarFeautureTypes
        start_row: int
                   Starting row relative to the original image
        start_col: int
                   Starting column relative to the original image
        w: int
                   Width of the whole feauture rectange(black and white)
        h: int 
                   Height of the whole feauture rectangle(black and white)

        Returns:
        feauture_value: int
                        value of the feauture (white - black)

        """
        black = 0
        white = 0
        if feauture_type == HaarFeautureTypes.TWO_HORIZONTAL:
            black = self.get_sum_in_rectangle(
                integral_image, start_row, start_col, w, h/2)
            white = self.get_sum_in_rectangle(
                integral_image, start_row+h/2, start_col, w, h/2)
        elif feauture_type == HaarFeautureTypes.TWO_VERTICAL:
            white = self.get_sum_in_rectangle(
                integral_image, start_row, start_col, w/2, h)
            black = self.get_sum_in_rectangle(
                integral_image, start_row, start_col + w/2, w/2, h)
        elif feauture_type == HaarFeautureTypes.THREE_HORIZONTAL:
            white = self.get_sum_in_rectangle(
                integral_image, start_row, start_col, w, h/3) + \
                self.get_sum_in_rectangle(
                    integral_image, start_row+(2*h/3), start_col, w, h/3)
            black = self.get_sum_in_rectangle(
                integral_image, start_row+(h/3), start_col, w, h/3)
        elif feauture_type == HaarFeautureTypes.THREE_VERTICAL:
            white = self.get_sum_in_rectangle(
                integral_image, start_row, start_col, w/3, h) + \
                self.get_sum_in_rectangle(
                    integral_image, start_row, start_col+(2*w/3), w/3, h)
            black = self.get_sum_in_rectangle(
                integral_image, start_row, start_col+(w/3), w/3, h)
        elif feauture_type == HaarFeautureTypes.FOUR_DIAGONAL:
            white = self.get_sum_in_rectangle(
                integral_image, start_row, start_col, w/2, h/2) + \
                self.get_sum_in_rectangle(
                    integral_image, start_row+h/2, start_col+w/2, w/2, h/2)
            black = self.get_sum_in_rectangle(
                integral_image, start_row+w/2, start_col, w/2, h/2) + \
                self.get_sum_in_rectangle(
                    integral_image, start_row+h/2, start_col, w/2, h/2)

        feauture_value = white-black
        return feauture_value

    def extract_features(self, original_image, width, height):
        """
        Extracts the features from the sliding window that slides over the 
        original image

        Parameters:
        original_image: np.array
        width: int
              width of the detection window
        height: int
              height of the detection window

        Returns:
        features_values: np.array
            contanins the extracted features values 
        """

        features = np.array([]).reshape(0, 5)
        for haar_type in range(len(HaarFeautureTypes)):
            wndx, wndy = self.haar_window[haar_type]
            for w in range(wndx, width+1, wndx):
                for h in range(wndy, height+1, wndy):
                    for x in range(0, width-w+1):
                        for y in range(0, height-h+1):
                            features = np.append(
                                features, [[haar_type, x, y, w, h]], axis=0)

        features_values = np.zeros((len(features)))
        integral_image = self.utils.get_integral_image(original_image)
        for i in range(len(features)):
            features_values[i] = self.get_feauture_value(
                integral_image,
                HaarFeautureTypes(features[i][0]),
                features[i][1],
                features[i][2],
                features[i][3],
                features[i][4]
            )

        return features_values


# utils = Utils()
# integral_image = utils.get_integral_image(
#     np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
# print(integral_image)
haar_like_feautures = HaarLikefeatures()
# print(feauture.get_sum_in_rectangle(integral_image, 0, 0, 2, 2))
print(haar_like_feautures.extract_features(
    np.array([[8, 2, 3, 7, 4, 9, 9, 10], [4, 5, 6, 1, 1, 3, 5, 6], [7, 8, 9, 8, 9, 8, 9, 8], [7, 8, 9, 8, 9, 8, 1, 9]]), 3, 3))
