import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('results.csv')

# 1) Scatter plot: Total BRAM vs Accuracy
plt.figure()
plt.scatter(df['brams'], df['accuracy'], s=1)
plt.xlabel('Total BRAMs')
plt.ylabel('Accuracy')
plt.title('Total BRAM vs Accuracy')
plt.show()