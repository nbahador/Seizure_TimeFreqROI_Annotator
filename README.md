## Seizure Time-Frequency ROI Annotator

This Python package provides tools to annotate spectrogram images of seizure episodes.

📦 **PyPI**: [seizure-timefreqroi-annotator](https://pypi.org/project/seizure-timefreqroi-annotator/)

---

### 📦 Installation

Install the package using pip:

```bash
pip install seizure-timefreqroi-annotator
```
---

### Annotation Interface Preview

<p>
  <img src="Seizure_TimeFreqROI_Annotator/Img/Ann_Gui.png" alt="Annotation GUI 1" width="49%" style="display: inline-block; border: 1px solid #ccc; padding: 4px; margin-right: 1%;">
  <img src="Seizure_TimeFreqROI_Annotator/Img/Ann_Gui2.png" alt="Annotation GUI 2" width="49%" style="display: inline-block; border: 1px solid #ccc; padding: 4px;">
</p>

---

### ▶️ Run Annotation GUI (Quick Start)
You can run the annotation tool directly from the command line using this one-liner (replace the paths with your local installed package path):

```python
python -c "from Seizure_TimeFreqROI_Annotator.datasets import SpectrogramDataset; from Seizure_TimeFreqROI_Annotator.annotator import collect_annotations; dataset = SpectrogramDataset(r'C:\Users\Nooshin\myenv\Lib\site-packages\Seizure_TimeFreqROI_Annotator\assets\sample_spectrograms'); print(f'Loaded {len(dataset.image_files)} images'); collect_annotations(dataset, r'C:\Users\Nooshin\myenv\Lib\site-packages\Seizure_TimeFreqROI_Annotator\assets\labels\annotations.xlsx'); print('Annotations saved')"
```

---
### ▶️ Run Annotation GUI (Quick Start)
You can run the annotation tool directly from the command line using this one-liner (replace the paths with your local installed package path):

```python
python -c "
from Seizure_TimeFreqROI_Annotator.datasets import SpectrogramDataset;
from Seizure_TimeFreqROI_Annotator.annotator import collect_annotations;
import os

# Define paths
spectrogram_dir = r'C:\Users\Nooshin\myenv\Lib\site-packages\Seizure_TimeFreqROI_Annotator\assets\sample_spectrograms'
output_file = r'C:\Users\Nooshin\myenv\Lib\site-packages\Seizure_TimeFreqROI_Annotator\assets\labels\annotations.xlsx'

# Load dataset and collect annotations
dataset = SpectrogramDataset(spectrogram_dir)
print(f'Successfully loaded {len(dataset.image_files)} spectrogram images')
collect_annotations(dataset, output_file)
print(f'Annotations saved to {os.path.basename(output_file)}')
"
```

---

