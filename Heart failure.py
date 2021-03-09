import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv('C:/Users/pishu/Downloads/727551_1263738_bundle_archive/heart_failure_clinical_records_dataset.csv')
print(df.head())
print(df.isnull().sum(),'\n\n')
corr=df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,cmap='coolwarm')
plt.show()
# Hence we know check which of the variables have absolute value of correlation greater than 0.15 with the death event variable
print(corr[abs(corr['DEATH_EVENT'])>0.15]['DEATH_EVENT'])
x=df[['age','ejection_fraction','serum_creatinine','time','serum_sodium']]
y=df['DEATH_EVENT']
accuracy={}
# we de\ivide the given dataset into training set and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2698)
# 1) Logistic Regression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
logreg_pred=logreg.predict(x_test)
accuracy['Logistic Regression']=accuracy_score(y_test,logreg_pred)
print('\n',classification_report(y_test,logreg_pred))
# 2) Support Vector Classifier
svm=SVC()
svm.fit(x_train,y_train)
svm_pred=svm.predict(x_test)
accuracy['SVC']=accuracy_score(y_test,svm_pred)
print('\n',classification_report(y_test,svm_pred))
# 3) K nearest neighbors with hyperparameter testing of the number of nearest neighbors to be considered
n_neighbours= list(range(1,10))
knn=KNeighborsClassifier()
params=dict(n_neighbors=n_neighbours)
knn_2=GridSearchCV(knn,params,cv=5)
knn_2.fit(x_train,y_train)
knn_2_pred=knn_2.predict(x_test)
accuracy['K Nearest Neighbors']=accuracy_score(y_test,knn_2_pred)
print('\n',classification_report(y_test,knn_2_pred))
# 4) Decision Tree Classifier 
dt=DecisionTreeClassifier(max_leaf_nodes=10,random_state=30,criterion='entropy')
dt.fit(x_train,y_train)
dt_pred=dt.predict(x_test)
accuracy['Decision Tree Classifier']=accuracy_score(y_test,dt_pred)
print('\n',classification_report(y_test,dt_pred))
#5) Random Forest Classifier with hyperparameter testing for the max_depth
max_depth=list(range(1,50))
params=dict(max_depth=max_depth)
fr_1=RandomForestClassifier(max_features=0.5,random_state=1)
fr=GridSearchCV(fr_1,params,cv=5)
fr.fit(x_train,y_train)
fr_pred=fr.predict(x_test)
accuracy['Random Forest Classifier']=accuracy_score(y_test,fr_pred)
print('\n',classification_report(y_test,fr_pred))
print(accuracy)




