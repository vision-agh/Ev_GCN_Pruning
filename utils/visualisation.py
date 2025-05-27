import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('results_mnist.csv')

# 1) Scatter plot: Total BRAM vs Accuracy
plt.figure()
plt.scatter(df['brams'], df['accuracy'], s=2)
plt.xlabel('Total BRAMs')
plt.ylabel('Accuracy')
plt.title('Total BRAM vs Accuracy')
plt.show()


# 3d plot BRAM vs M vs Accuracy

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['brams'], df['m'], df['accuracy'], s=2)
# ax.set_xlabel('Total BRAMs')
# ax.set_ylabel('Total Multipliers')
# ax.set_zlabel('Accuracy')
# 
# plt.title('Total BRAM vs Total Multipliers vs Accuracy')
# plt.show()