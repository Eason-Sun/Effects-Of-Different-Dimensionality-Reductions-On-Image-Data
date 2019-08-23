import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('DataB.csv', index_col=0)
class_matrix = df['gnd'].values
df.drop(columns=['gnd'],inplace=True)
data_matrix = df.astype(float).values

data_matrix_std = StandardScaler().fit_transform(data_matrix)  # Standardization of data matrix.

lda = LinearDiscriminantAnalysis(n_components=4)  # Apply LDA with 4 components
projected_matrix = lda.fit_transform(data_matrix_std, class_matrix)  # Train LDA with all data samples

sns.set(font_scale=1.2)

lda_df = pd.DataFrame(projected_matrix, columns=['1st Component', '2nd Component','3rd Component','4th Component'])
lda_df['Classes'] = class_matrix

gnb = GaussianNB()  # Use Gaussian Naive Bayes' Classifier.

# Split the data matrix randomly to 70% training data and 30% of testing data, then train for 5 iterations.
for i in range(5):
    feature_train, feature_test, target_train, target_test = train_test_split(lda_df.iloc[:,0:4], class_matrix, test_size=0.3)
    gnb.partial_fit(feature_train, target_train, classes=np.unique(target_train))
    s = accuracy_score(target_test, gnb.predict(feature_test))
    print(s)

# Plot the projected samples in different colours by their classes.
sns.lmplot(x="1st Component", y="2nd Component",
  data=lda_df,
  fit_reg=False,
  hue='Classes', # color by Classes
  legend=True,
  scatter_kws={"s": 80},
  )
plt.suptitle('Linear Discriminant Analysis')
plt.show()