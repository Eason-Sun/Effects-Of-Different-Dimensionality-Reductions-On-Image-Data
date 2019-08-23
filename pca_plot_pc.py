import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('DataB.csv', index_col=0)
class_matrix = df['gnd'].values
df.drop(columns=['gnd'],inplace=True)
data_matrix = df.astype(float).values

data_matrix_std = StandardScaler().fit_transform(data_matrix)  # Standardization of data matrix.
covariance_matrix = np.cov(data_matrix_std.T)  # Find the covariance matrix.
eig_val, eig_vec = np.linalg.eig(covariance_matrix)  # Compute the eigenvalues and eigenvectors.

# Sort the list of (eigenvalue, eigenvector) by the absolute eigenvalue in descending order.
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort()
eig_pairs.reverse()

sns.set(font_scale=1.2)

# Compute the projected data matrix by the first two principle components.
transformation_matrix_pc_1_2 = np.hstack((eig_pairs[0][1].reshape(len(eig_val),1), eig_pairs[1][1].reshape(len(eig_val),1)))
pc_1_2_matrix = data_matrix_std.dot(transformation_matrix_pc_1_2)
pc_1_2_df = pd.DataFrame(data=pc_1_2_matrix,columns=['PC1', 'PC2'])
pc_1_2_df['Classes'] = class_matrix

# Plot the projected samples in different colours by their classes.
sns.lmplot(x="PC1", y="PC2",
  data=pc_1_2_df,
  fit_reg=False,
  hue='Classes',
  legend=True,
  scatter_kws={"s": 80},
  )
plt.axvline(x=-20,color='#d62728')
plt.axvline(x=10,color='#d62728')
plt.axhline(y=-15,color='#1f77b4')
plt.axhline(y=10,color='#1f77b4')

# Compute the projected data matrix by the 5th and 6th principle components.
transformation_matrix_pc_5_6 = np.hstack((eig_pairs[4][1].reshape(len(eig_val),1), eig_pairs[5][1].reshape(len(eig_val),1)))
pc_5_6_matrix = data_matrix_std.dot(transformation_matrix_pc_5_6)
pc_5_6_df = pd.DataFrame(data=pc_5_6_matrix,columns=['PC5', 'PC6'])
pc_5_6_df['Classes'] = class_matrix

# Plot the projected samples in different colours by their classes.
sns.lmplot(x="PC5", y="PC6",
  data=pc_5_6_df,
  fit_reg=False,
  hue='Classes', # color by Classes
  legend=True,
  scatter_kws={"s": 80})
plt.axvline(x=-12,color='#d62728')
plt.axvline(x=11,color='#d62728')
plt.axhline(y=-8,color='#1f77b4')
plt.axhline(y=10,color='#1f77b4')

plt.show()

