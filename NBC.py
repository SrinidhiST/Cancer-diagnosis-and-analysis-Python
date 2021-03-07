#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#NBC CLASSIFIER
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
print_scores(nb, x_train, y_train, 10, 'accuracy')
nb.fit(x_train,y_train)
y_pred_nb = nb.predict(x_test)
print('Classification Report', classification_report(y_test, y_pred_nb))
print('Confusion Matrix', confusion_matrix(y_test, y_pred_nb))
print("Training Accuracy", nb.score(x_train, y_train)*100)
print("Testing Accuracy", nb.score(x_test, y_test)*100)
score_nb = round(accuracy_score(y_pred_nb,y_test)*100,2)
print("The accuracy score achieved using Naive Bayes Model is: "+str(score_nb)+" %")
model = GaussianNB()
from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)

