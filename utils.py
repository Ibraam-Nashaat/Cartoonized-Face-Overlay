import numpy as np
import os
import pickle
from skimage import io
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool
import time
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
    
    def read_pgm_dataset(self, folder_path):
        """
        Read the images from the dataset folder
        Parameters:
        folder_path: str
            path to the folder containing the dataset
        Returns:
        images: np.array
            numpy array of images
        """

        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pgm"):
                file_path = os.path.join(folder_path, filename)
                img = io.imread(file_path)

                images.append(img)

        return np.array(images)
    
    def save_pickle(self, obj, file_path):
        """
        Save an object to a file using pickle

        Parameters:
        obj: object
        file_path: str
            path to the file
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            print(e)
            print("Error saving pickle file")

    def load_pickle(self, file_path):
        """
        Load an object from a file using pickle
        Parameters:
        file_path: str
            path to the file
        Returns:
        obj: object
        """
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            print(e)
            print("Error loading pickle file")

        return obj
    
    def display_random_images(self, images, n=5, figsize=(10, 10), dpi=100):
        """
        Display random images from the dataset
        Parameters:
        images: np.array
            numpy array of images
        n: int
            number of images to display
        figsize: tuple
            figure size
            default: (10, 10)
        dpi: int
            figure dpi
            default: 100
        """
        fig, axes = plt.subplots(1, n, figsize=figsize)
        for i in range(n):
            index = random.randint(0, len(images))
            axes[i].imshow(images[index], cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    def apply_func_parallel_nump_array(self, func, array, n_process, *args):
        """
        Apply a function to a numpy array in parallel
        Parameters:
        func: function
            function to apply to the array
        array: np.array
            numpy array
        n_process: int
            number of process to run in parallel
            default: 8
        *args: tuple
            arguments to pass to the function
        Returns:
        result: np.array
            numpy array of the result
        """
        params = [(i, *args) for i in array]
        start_time = time.time()
        print("Start parallel processing")
        with Pool(processes=n_process) as pool:
            result = pool.starmap(func, params)
            
        print(f"Parallel processing finished in {time.time()-start_time} seconds")
        
        return np.array(result)