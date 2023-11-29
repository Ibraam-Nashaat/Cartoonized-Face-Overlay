import numpy as np


class Utils:
    def get_integral_image(self, original_image):
        """
        Get the integral image from the original image.

        original image        =>         integral image
        [[1, 2, 3]                       [[ 0.  0.  0.  0.]
         [4, 5, 6]                        [ 0.  1.  3.  6.]
         [7, 8, 9]]                       [ 0.  5. 12. 21.]
                                          [ 0. 12. 27. 45.]]
        Parameters:
        original_image: np.array

        Returns: 
        integral_image: np.array

        """
        rows, columns = original_image.shape
        integral_image = np.zeros((rows+1, columns+1))
        for i in range(1, rows+1):
            for j in range(1, columns+1):
                integral_image[i, j] = integral_image[i-1, j]+integral_image[i,
                                                                             j-1]-integral_image[i-1, j-1]+original_image[i-1, j-1]

        return integral_image


# utils = Utils()
# # print(utils.get_integral_image(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
