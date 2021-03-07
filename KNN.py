#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    #Fit the model
    knn.fit(x_train, y_train)  
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)
        #Compute accuracy on the test set
    test_accuracy[i] = knn.score(x_test, y_test)
print_scores(knn, x_train, y_train, 10, 'accuracy')
plt.title('KNN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)
print('Classification Report', classification_report(y_test, y_pred_knn))
print('Confusion Matrix', confusion_matrix(y_test, y_pred_knn))
print("Training Accuracy", knn.score(x_train, y_train)*100)
print("Testing Accuracy", knn.score(x_test, y_test)*100)
score_knn = round(accuracy_score(y_pred_knn,y_test)*100,2)
print("The accuracy score achieved using KNN Model is: "+str(score_knn)+" %")

#USING HYPER-PARAMETER TUNING
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
df = pd.read_csv(r"C:\Users\Divya Haridas\Desktop\cancer.csv")
df.head()
x = df.iloc[:, :-1].values
print(x)
y = df.iloc[:, -1].values
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42, stratify=y)
from sklearn.metrics import scorer
from time import time
knn = KNeighborsClassifier()
params = {'n_neighbors':range(2,11),'metric':['minkowski','manhattan','euclidean']}
grid_search = GridSearchCV(knn,param_grid = params,scoring='precision',cv=2)
grid_search.fit(x_train,y_train)
print("Grid Search Results\n \n")
print("Best Params : ",grid_search.best_params_)
print("Best Precision : ",grid_search.best_score_)
print("Best Model: \n",grid_search.best_estimator_)
knn = KNeighborsClassifier()
params = {'n_neighbors':np.arange(2,11,step=1),'metric':['minkowski','manhattan','euclidean']}
random_search = RandomizedSearchCV(knn,params,scoring='precision' ,cv=2)
random_search.fit(x_train,y_train)
print("Random Search Results")
print("Best Params : ",random_search.best_params_)
print("Best Precision : ",random_search.best_score_)
print("Best Model: \n",random_search.best_estimator_)
tuned_model = random_search.best_estimator_
y_pred = tuned_model.predict(x_test).flatten()
print("Fine Tuned Model results\n\n")
conf_mat=confusion_matrix(y_test,y_pred)
print("Confusion Matrix \n",conf_mat)
print("\nClassification Report : \n",classification_report(y_test,y_pred))
tuned_accuracy = (conf_mat[0][0]+conf_mat[1][1])/len(y_test)
print("Accuracy : " ,tuned_accuracy )
probs=tuned_model.predict_proba(x_test)
probs=probs[:,1]
fpr,tpr,_=roc_curve(y_test,probs)
random_probs = [0 for _ in range(len(y_test))]
p_fpr,p_tpr,_ = roc_curve(y_test,random_probs)
auc_score=roc_auc_score(y_test,probs)
print("AUC SCORE : " ,auc_score)
plt.plot(p_fpr, p_tpr, linestyle='--')
plt.plot(fpr, tpr, marker='.', label='TUNED KNN (area=%0.2f)'% auc_score)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC-AUC CURVE for TUNED KNN Classifier")
plt.legend()
plt.show()

