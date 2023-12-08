
class Errors:
    def __init__(self):
        self.empty_image = "Original image array is empty"
        self.window_out_of_bounds = "Window is out of original image's bounds"
        self.negative_dimensions = "One or both of the window dimensions is zero or negative"

    def get_empty_image_message(self):
        return self.empty_image

    def get_window_out_of_bounds_message(self):
        return self.window_out_of_bounds

    def get_negative_dimensions_message(self):
        return self.negative_dimensions
