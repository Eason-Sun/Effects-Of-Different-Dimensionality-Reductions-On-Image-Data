import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('DataB.csv', index_col=0)
class_matrix = df['gnd'].values
df.drop(columns=['gnd'],inplace=True)
data_matrix = df.astype(float).values

# Compute the first four PCA components and the projected data matrix.
data_matrix_std = StandardScaler().fit_transform(data_matrix)
covariance_matrix = np.cov(data_matrix_std.T)
eig_val, eig_vec = np.linalg.eig(covariance_matrix)
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort()
eig_pairs.reverse()
transformation_matrix = np.hstack((eig_pairs[0][1].reshape(len(eig_val), 1),eig_pairs[1][1].reshape(len(eig_val), 1),eig_pairs[2][1].reshape(len(eig_val), 1),eig_pairs[3][1].reshape(len(eig_val), 1)))
pc_matrix = data_matrix_std.dot(transformation_matrix)

gnb = GaussianNB()  # Use Gaussian Naive Bayes' Classifier.

# Split the data matrix randomly to 70% training data and 30% of testing data, then train for 5 iterations.
for i in range(5):
    feature_train, feature_test, target_train, target_test = train_test_split(pc_matrix, class_matrix, test_size=0.3)
    gnb.partial_fit(feature_train, target_train, classes=np.unique(target_train))
    s = accuracy_score(target_test, gnb.predict(feature_test))
    print(s)  # Print the accuracy for each training iteration.

num_of_pc_list = [2,4,10,30,60,200,500,784]
classification_errors = []

# Find the classification error for different number of principle components.
for num_of_pc in num_of_pc_list:
    pc_tup = ()
    for j in range(num_of_pc):
        pc_tup = pc_tup + (eig_pairs[j][1].reshape(len(eig_val), 1),)
    transformation_matrix = np.hstack(pc_tup)
    pc_matrix = data_matrix_std.dot(transformation_matrix)
    pc_df = pd.DataFrame(data=pc_matrix)
    pc_df['Classes'] = class_matrix
    gnb = GaussianNB()
    pred_result = gnb.fit(pc_df.iloc[:, 0:num_of_pc], pc_df['Classes']).predict(pc_df.iloc[:, 0:num_of_pc])
    classification_error = (pred_result != pc_df['Classes']).sum() / pred_result.shape[0]
    classification_errors.append(classification_error)

# Plot the classification error vs number of principle components.
plt.plot(num_of_pc_list, classification_errors,'-ok')
plt.suptitle('Classification Error vs Number of Principle Components', fontsize=20)
plt.ylabel('Classification Error', fontsize=15)
plt.xlabel('Number of Principle Components', fontsize=15)
plt.show()