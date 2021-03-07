#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#SVM
dt = pd.read_excel('C:/Users/SRS/Downloads/cancer patient data sets.xlsx')
x=dt.iloc[:,1].values.reshape(-1,1)
y=dt.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.15,random_state=101)
#using linear kernel
from sklearn import svm
sv = svm.SVC(kernel='linear')
print_scores(sv, X_train, Y_train, 10, 'accuracy')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print('Classification Report', classification_report(Y_test, Y_pred_svm))
print('Confusion Matrix', confusion_matrix(Y_test, Y_pred_svm))
print("Training Accuracy", sv.score(X_train, Y_train)*100)
print("Testing Accuracy", sv.score(X_test, Y_test)*100)
from sklearn.metrics import accuracy_score
score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM Model is: "+str(score_svm)+" %")
#using radial basis kernel
from sklearn import svm
sv = svm.SVC(kernel= 'rbf')
print_scores(sv, X_train, Y_train, 10, 'accuracy')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
print('Classification Report', classification_report(Y_test, Y_pred_svm))
print('Confusion Matrix', confusion_matrix(Y_test, Y_pred_svm))
print("Training Accuracy", sv.score(X_train, Y_train)*100)
print("Testing Accuracy", sv.score(X_test, Y_test)*100)
score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM Model is: "+str(score_svm)+" %")
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix 
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],              'kernel': ['rbf']}  
  grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split( X,Y, test_size = 0.30, random_state = 101) 
  # fitting the model for grid search 
grid.fit(X_train, Y_train) 
# print best parameter after tuning 
print(grid.best_params_) 
  # print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
grid_predictions = grid.predict(X_test) 
# print classification report 
print(classification_report(Y_test, grid_predictions))

