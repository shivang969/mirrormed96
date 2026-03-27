import cv2
import numpy as np

def remove_hair(image_path):
    """
    Applies the DullRazor algorithm to digitally remove hair from skin lesions.
    """
    # 1. Load image and convert to RGB (OpenCV uses BGR by default)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Convert to grayscale for morphological operations
    gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 3. Apply morphological black-hat filtering to isolate dark, thin structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    
    # 4. Intensify the hair contours
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 5. Inpaint the masked area (replace hair pixels with surrounding skin colors)
    inpainted_image = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    
    return inpainted_image

# You can test it by un-commenting the lines below once your data is downloaded:
# if __name__ == "__main__":
#     cleaned = remove_hair("../data/HAM10000_images_part_1/ISIC_0024306.jpg")
#     cv2.imwrite("test_cleaned.jpg", cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))
#     print("Test image saved as test_cleaned.jpg")