import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('DataB.csv', index_col=0)
class_matrix = df['gnd']
df.drop(columns=['gnd'], inplace=True)
n_neighbors = 5

# Apply ISOMAP to the dataset.
print("Training ISOMAP...")
iso = manifold.Isomap(n_neighbors, n_components=4)
iso.fit(df)
X_iso = iso.transform(df)

gnb = GaussianNB()  # Use Gaussian Naive Bayes' Classifier.
iso_df = pd.DataFrame(data=X_iso, columns=['Comp_1','Comp_2','Comp_3','Comp_4'], index=(np.arange(1,len(X_iso)+1)))
# Split the data matrix randomly to 70% training data and 30% of testing data, then train for 5 iterations.
print("Classification accuracy of ISOMAP in 5 iterations:")
for i in range(5):
    feature_train, feature_test, target_train, target_test = train_test_split(iso_df, class_matrix, test_size=0.3)
    gnb.partial_fit(feature_train, target_train, classes=np.unique(target_train))
    s = accuracy_score(target_test, gnb.predict(feature_test))
    print(s)


# Apply LLE to the dataset.
print("\nTraining LLE...")
lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components=4, method='standard')
lle.fit(df)
X_lle = lle.transform(df)

gnb = GaussianNB()  # Use Gaussian Naive Bayes' Classifier.
lle_df = pd.DataFrame(data=X_lle, columns=['Comp_1','Comp_2','Comp_3','Comp_4'], index=(np.arange(1,len(X_lle)+1)))
# Split the data matrix randomly to 70% training data and 30% of testing data, then train for 5 iterations.
print("Classification accuracy of ISOMAP in 5 iterations:")
for i in range(5):
    feature_train, feature_test, target_train, target_test = train_test_split(lle_df, class_matrix, test_size=0.3)
    gnb.partial_fit(feature_train, target_train, classes=np.unique(target_train))
    s = accuracy_score(target_test, gnb.predict(feature_test))
    print(s)

