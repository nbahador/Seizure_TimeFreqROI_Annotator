import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import pandas as pd

class BaseSpectrogramDataset:
    def __init__(self, image_dir: str):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
        self.image_dir = image_dir
        self.image_files = self._get_image_files()
        
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in directory: {image_dir}")
            
        self.annotations = []
        self.no_roi_images = []
        
    def _get_image_files(self) -> List[str]:
        """Get list of image files in directory"""
        return [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    def get_annotation_samples(self, sample_size: Optional[int] = None) -> List[str]:
        """Return a list of image files to annotate"""
        if sample_size and sample_size < len(self.image_files):
            return np.random.choice(self.image_files, size=sample_size, replace=False)
        return self.image_files.copy()
    
    def mark_image_as_no_roi(self, image_file: str) -> None:
        """Mark an image as having no ROI"""
        if image_file not in self.no_roi_images:
            self.no_roi_images.append(image_file)
    
    def get_patch_indices_in_roi(self, image_file: str, x1: int, y1: int, 
                                x2: int, y2: int) -> List[Tuple[int, int, int, int]]:
        """Get indices of patches within the ROI"""
        return [(x1, y1, x2, y2)]
    
    def annotate_roi_patches(self, patch_indices: List[Tuple[int, int, int, int]]) -> None:
        """Store the annotated patch indices"""
        self.annotations.extend(patch_indices)
    
    def save_annotations_to_csv(self, output_path: str) -> None:
        """Save annotations to CSV file"""
        df = pd.DataFrame(self.annotations)
        df.to_csv(output_path, index=False)