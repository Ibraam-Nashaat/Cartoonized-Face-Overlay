import skimage.io as io
from skimage.color import rgb2gray
from utils import *
from skimage.transform import integral_image as sk_integral_image


def test_integral_image():
    img = io.imread("Images/hermoine.jpg")
    img_gray = (rgb2gray(img)*255).astype('uint8')
    utils = Utils()
    custom_integral_image = np.array(
        utils.get_integral_image(img_gray)[1:, 1:], dtype=int)

    sk_integral = np.array(sk_integral_image(img_gray), dtype=int)

    assert np.array_equal(custom_integral_image,
                          sk_integral), "The integral images do not match!"
