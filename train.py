"""
Data source: https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv
"""



import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("Iris.csv")
y = df.pop('Species')
# Split into train and test sections

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

df['Species'] = y

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
clf = RandomForestClassifier(max_depth=2, random_state=seed)
clf.fit(X_train, y_train)

# Report training set score
train_score = clf.score(X_train, y_train) * 100
# Report test set score
test_score = clf.score(X_test, y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)


##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################

# Calculate feature importance in random forest
importances = clf.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()


##########################################
############ PLOT CONF MATRIX  ###########
##########################################

y_pred = clf.predict(X_test)
res_df = pd.DataFrame(list(zip(y_test,y_pred)), columns = ["true","pred"])


conf_matrix = confusion_matrix(y_test,y_pred)

display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Iris-setosa', 'Iris-versicolor','Iris-virginica'])

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

display.plot()
plt.savefig('confusion_matrix.png',dpi=120)
plt.close()