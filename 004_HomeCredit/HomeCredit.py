import numpy as np
import pandas as pd
import sys
sys.path.append('../')

# Import data
# Data is filled from the CSV file
train_base = pd.read_csv("../../HomeCredit/csv_files/train/train_base.csv")

print(train_base.shape)

print(train_base.head())