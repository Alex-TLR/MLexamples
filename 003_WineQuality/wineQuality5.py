import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import seaborn as sb 
import matplotlib.pyplot as plt

# Correlogram

data = pd.read_csv("winequality-red.csv")

print("Data info")
print(data.head())
print("\n")

plt.figure(figsize=(12, 12))
sb.heatmap(data.corr() > 0.6, annot=True, cbar=False)
plt.show()

data = data.drop('fixed acidity', axis=1)
data = data.drop('free sulfur dioxide', axis=1)
print(data.corr())

sb.pairplot(data)
plt.show()
