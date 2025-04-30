from setuptools import setup, find_packages


setup(
    name="seizure-timefreqroi-annotator",  # PyPI prefers lowercase with hyphens
    version="0.1.0",
    author="Nooshin Bahador",
    author_email="nooshin.bah@gmail.com",
    long_description="A package for annotation of ROIs on seizure spectrograms",
    long_description_content_type="text/plain",
    url="https://github.com/nbahador/Seizure_TimeFreqROI_Annotator",
    project_urls={
        "Bug Tracker": "https://github.com/nbahador/Seizure_TimeFreqROI_Annotator/issues",
        "Documentation": "https://github.com/nbahador/Seizure_TimeFreqROI_Annotator#readme",
    },
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'Pillow',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'transformers',
        'peft',
        'openpyxl',
        'scipy',
        'tqdm',
        'pyyaml',
    ],
    python_requires='>=3.8',
    license="MIT",
    keywords=[
        "seizure",
        "spectrogram",
        "annotation",
        "medical",
        "neuroscience",
        "time-frequency",
        "ROI"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)