from .core import ROISingleImageAnnotator 
from typing import Optional
from ..datasets import SpectrogramDataset
import matplotlib.pyplot as plt

def collect_annotations(dataset: SpectrogramDataset, output_excel: str, 
                       sample_size: Optional[int] = None) -> None:
    """Main function to collect annotations from user"""
    print("\nStarting annotation...")
    annotator = ROISingleImageAnnotator(dataset, output_excel, sample_size)
    annotator.show_next_image()
    
    while annotator.is_active or not annotator.annotation_complete:
        plt.pause(0.1)
        if not plt.fignum_exists(annotator.fig.number) if annotator.fig else False:
            if not annotator.annotation_complete:
                annotator.is_active = False
                annotator.annotation_complete = True
            break
    
    print(f"\nAnnotation complete! Processed {annotator.current_img_idx} images.")
    plt.close('all')