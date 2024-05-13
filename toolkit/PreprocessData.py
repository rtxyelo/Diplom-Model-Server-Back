import cv2
import os

class Preprocessing:
    """Class for image preprocessing."""
    
    blur_value_threshold = (1, 3)
    exposure_alpha_threshold = (0.3, 1.5)
    exposure_beta_threshold = (-100, 100)

    def __init__(self, target_size, blur_value=0, exposure_values=[1.0, 0]):
        """Initialize preprocessing instance."""
        self.target_size = target_size
        self.blur_value = int(blur_value)  
        self.exposure_values = (float(exposure_values[0]), int(exposure_values[1]))

    def adjust_exposure(self, image, alpha=1.0, beta=0):
        """Adjusts the exposure of the image using alpha (contrast) and beta (brightness) parameters."""
        alpha = min(max(alpha, self.exposure_alpha_threshold[0]), self.exposure_alpha_threshold[1])
        beta = min(max(beta, self.exposure_beta_threshold[0]), self.exposure_beta_threshold[1])
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted_image

    def preprocess_image(self, image, save=True):
        """Preprocessing image."""
        if image is None:
            raise ValueError("Image not provided.")
        
        # Resize the image
        dim = (self.target_size, self.target_size)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Blur image
        if (self.blur_value != 0):
            self.blur_value = min(max(self.blur_value, self.blur_value_threshold[0]), self.blur_value_threshold[1])
            if (self.blur_value % 2 == 0):
                self.blur_value += 1 
            blur = (self.blur_value, self.blur_value)
            #self.blur_image = cv2.blur(gray_image, blur)
            blur_image = cv2.GaussianBlur(gray_image, blur, 0)
        else:
            blur_image = gray_image

        # Exposure image
        exposured_image = self.adjust_exposure(blur_image, self.exposure_values[0], self.exposure_values[1])

        # Result image
        self.result_image = exposured_image

        if save:
            self.save_to_dir()

        return self.result_image

    def save_to_dir(self, image_path='C:/Users/greyb/OneDrive/Desktop/Potok/Study/Diplom/test/test/prepr_imgs', image_name='prepr_img', image_format='jpg'):
        """Save preprocessed image to directory"""
        cv2.imwrite(os.path.join(image_path, f'{image_name}.{image_format}'), self.result_image)