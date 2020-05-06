#Sai Vivek Amirishetty- https://github.com/vivekboss99/Red-Wine-Quality-Prediction.git -for dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
import numpy as np

wine =pd.read_csv('winequality-red.csv')
#wine.head()

bins=(2,6.5,8)
group_names=['bad','good']
wine['quality']=pd.cut(wine['quality'],bins=bins,labels=group_names)
label_quality=LabelEncoder()
wine['quality']=label_quality.fit_transform(wine['quality'])
X=wine.drop('quality',axis=1)
y=wine['quality']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)

rfc_eval=cross_val_score(estimator=rfc,X=X_train,y=y_train,cv=10)
print(rfc_eval.mean())
