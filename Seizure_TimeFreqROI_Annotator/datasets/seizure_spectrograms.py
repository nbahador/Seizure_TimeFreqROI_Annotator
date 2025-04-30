import numpy as np
from PIL import Image
from typing import Tuple
from .base import BaseSpectrogramDataset

class SpectrogramDataset(BaseSpectrogramDataset):
    def __init__(self, image_dir: str, freq_range: Tuple[float, float] = (1, 60),
                 time_range: Tuple[float, float] = (0, 60)):
        super().__init__(image_dir)
        self.freq_range = freq_range
        self.time_range = time_range
        
    def _preprocess_image(self, img_array: np.ndarray) -> Tuple[np.ndarray, int]:
        """Clean the image by removing white borders and keeping only the spectrogram"""
        threshold = 240  # Adjust based on your images
        binary = img_array < threshold
        
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        cleaned_img = img_array[rmin:rmax+1, cmin:cmax+1]
        spectrogram_height = rmax - rmin
        
        return cleaned_img, spectrogram_height
    
    def pixel_to_freq(self, y_pixel: float, spectrogram_height: int) -> float:
        """Convert y-pixel coordinate to frequency in Hz"""
        freq_min, freq_max = self.freq_range
        normalized_y = y_pixel / spectrogram_height
        return freq_max - normalized_y * (freq_max - freq_min)
    
    def pixel_to_time(self, x_pixel: float, image_width: int) -> float:
        """Convert x-pixel coordinate to time in seconds"""
        time_min, time_max = self.time_range
        return time_min + (x_pixel / image_width) * (time_max - time_min)