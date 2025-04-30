import numpy as np
from PIL import Image

def remove_white_border(image, threshold: int = 240):
    """Clean the image by removing white borders and keeping only the spectrogram"""
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    binary = img < threshold
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    img = img[rmin:rmax+1, cmin:cmax+1]
    
    if isinstance(image, Image.Image):
        return Image.fromarray(img)
    return img

def convert_to_grayscale(image):
    """Convert image to grayscale"""
    if isinstance(image, Image.Image):
        return image.convert('L')
    else:
        return Image.fromarray(image).convert('L')