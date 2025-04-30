import os
import numpy as np
import pandas as pd
from matplotlib.widgets import RectangleSelector, Button
import matplotlib.patches as patches
from typing import Optional
from ..datasets import SpectrogramDataset
from PIL import Image
import matplotlib.pyplot as plt

class ROISingleImageAnnotator:
    def __init__(self, dataset: SpectrogramDataset, output_excel: str, 
                 sample_size: Optional[int] = None):
        self.dataset = dataset
        self.current_img_idx = 0
        self.annotation_samples = dataset.get_annotation_samples(sample_size)
        self.fig = None
        self.ax = None
        self.current_rois = []
        self.is_active = True
        self.btn_no_roi = None
        self.btn_next = None
        self.btn_done = None
        self.roi_patches = []
        self.annotation_complete = False
        self.output_excel = output_excel
        self.annotations = []
        
    def onselect(self, eclick, erelease) -> None:
        x1, y2 = int(eclick.xdata), int(eclick.ydata)
        x2, y1 = int(erelease.xdata), int(erelease.ydata)
        
        img_file = self.annotation_samples[self.current_img_idx]
        img = np.array(Image.open(img_file).convert('L'))
        _, spectrogram_height = self.dataset._preprocess_image(img)
        
        start_time = self.dataset.pixel_to_time(x1, img.shape[1])
        end_time = self.dataset.pixel_to_time(x2, img.shape[1])
        start_freq = self.dataset.pixel_to_freq(y1, spectrogram_height)
        end_freq = self.dataset.pixel_to_freq(y2, spectrogram_height)
        
        roi = (x1, y1, x2, y2, start_time, end_time, start_freq, end_freq)
        self.current_rois.append(roi)
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)
        self.fig.canvas.draw()
        self.save_current_annotation()
    
    def save_current_annotation(self) -> None:
        """Save the current image's annotations to the list"""
        img_file = self.annotation_samples[self.current_img_idx]
        base_name = os.path.basename(img_file)
        
        if not self.current_rois:
            self.annotations.append({
                'image_file': base_name,
                'start_time': None,
                'end_time': None,
                'start_freq': None,
                'end_freq': None,
                'has_roi': False
            })
        else:
            for roi in self.current_rois:
                x1, y1, x2, y2, start_time, end_time, start_freq, end_freq = roi
                self.annotations.append({
                    'image_file': base_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_freq': start_freq,
                    'end_freq': end_freq,
                    'has_roi': True
                })
        
        self.save_annotations_to_excel()
    
    def show_next_image(self) -> bool:
        if not self.is_active or self.current_img_idx >= len(self.annotation_samples):
            self.is_active = False
            self.annotation_complete = True
            if self.fig is not None:
                plt.close(self.fig)
            self.save_annotations_to_excel()
            return False
            
        img_file = self.annotation_samples[self.current_img_idx]
        try:
            img = np.array(Image.open(img_file).convert('L'))
            img, spectrogram_height = self.dataset._preprocess_image(img)
            
            self.current_rois = []
            self.roi_patches = []
            
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
            
            self.fig = plt.figure(num=f"Annotation {self.current_img_idx+1}/{len(self.annotation_samples)}", 
                                figsize=(10, 8))
            plt.subplots_adjust(bottom=0.25)
            
            self.ax = self.fig.add_subplot(111)
            self.ax.imshow(img, cmap='gray', aspect='auto')
            self.ax.set_title(f"Image {self.current_img_idx+1}/{len(self.annotation_samples)}")
            
            self.rs = RectangleSelector(self.ax, self.onselect,
                                     useblit=True, button=[1],
                                     minspanx=5, minspany=5,
                                     spancoords='pixels', interactive=True)
            
            ax_no_roi = plt.axes([0.1, 0.05, 0.2, 0.075])
            ax_done = plt.axes([0.4, 0.05, 0.2, 0.075])
            ax_next = plt.axes([0.7, 0.05, 0.2, 0.075])
            
            self.btn_no_roi = Button(ax_no_roi, 'No ROI', color='lightgoldenrodyellow', hovercolor='0.975')
            self.btn_done = Button(ax_done, 'Done with ROIs', color='lightgreen', hovercolor='0.975')
            self.btn_next = Button(ax_next, 'Next Image', color='lightgoldenrodyellow', hovercolor='0.975')
            
            current_img_file = img_file
            
            def handle_no_roi(event):
                self.dataset.mark_image_as_no_roi(current_img_file)
                print(f"Marked {os.path.basename(current_img_file)} as no ROI")
                self.save_current_annotation()
                self.move_to_next_image()
                
            def handle_done(event):
                for roi in self.current_rois:
                    x1, y1, x2, y2, _, _, _, _ = roi
                    indices = self.dataset.get_patch_indices_in_roi(current_img_file, x1, y1, x2, y2)
                    self.roi_patches.extend(indices)
                
                if self.roi_patches:
                    self.dataset.annotate_roi_patches(self.roi_patches)
                    print(f"Annotated {len(self.roi_patches)} patches in {os.path.basename(current_img_file)}")
                
                self.current_rois = []
                self.roi_patches = []
                
            def handle_next(event):
                handle_done(None)
                self.move_to_next_image()
            
            self.btn_no_roi.on_clicked(handle_no_roi)
            self.btn_done.on_clicked(handle_done)
            self.btn_next.on_clicked(handle_next)
            
            self.fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.1)
            return True
        except Exception as e:
            print(f"Error loading image {img_file}: {str(e)}")
            self.current_img_idx += 1
            return self.show_next_image()
    
    def move_to_next_image(self) -> None:
        plt.close(self.fig)
        self.current_img_idx += 1
        if self.current_img_idx < len(self.annotation_samples):
            plt.pause(0.5)
            self.show_next_image()
        else:
            self.is_active = False
            self.annotation_complete = True
            self.save_annotations_to_excel()
    
    def save_annotations_to_excel(self) -> None:
        if not self.annotations:
            print("No annotations to save.")
            return
            
        df = pd.DataFrame(self.annotations)
        df = df[['image_file', 'has_roi', 'start_time', 'end_time', 'start_freq', 'end_freq']]
        
        try:
            os.makedirs(os.path.dirname(self.output_excel), exist_ok=True)
            df.to_excel(self.output_excel, index=False)
            print(f"Annotations saved to {os.path.abspath(self.output_excel)}")
        except Exception as e:
            print(f"Error saving annotations to Excel: {str(e)}")