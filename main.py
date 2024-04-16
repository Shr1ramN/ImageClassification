import numpy as np
import pandas as pd
from pathlib import Path
import os.path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

image_dir = Path('D:/Image Classification/Indian Food Images')

filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

# Count the number of samples for each category
category_counts = images['Label'].value_counts()

# Set the desired number of samples per category
samples_per_category = 100

category_samples = []
for category in images['Label'].unique():
    # Check if there are enough samples for this category
    if category_counts[category] >= samples_per_category:
        category_slice = images.query("Label == @category")
        category_samples.append(category_slice.sample(samples_per_category, random_state=1))
    else:
        # If there are fewer than 100 samples, sample all available samples
        print(f"Not enough samples for category '{category}'. Sampling all available samples...")
        category_samples.append(images.query("Label == @category"))  # Fix: initialize category_slice here

image_df = pd.concat(category_samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

image_df
