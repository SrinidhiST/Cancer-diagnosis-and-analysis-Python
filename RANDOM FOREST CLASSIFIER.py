#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier as RTC
classifier = RTC(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
rtc = RTC()
rtc.fit(x_train,y_train)
#MODEL PERFORMANCE EVALUATION
from sklearn.metrics import classification_report,confusion_matrix
x_test = np.nan_to_num(x_test)
y_pred = rtc.predict(x_test)
print(classification_report(y_test,y_pred))
conf_mat = confusion_matrix(y_test,y_pred)
print("Confusion Matrix: ",(conf_mat))
print("Accuracy : ",(conf_mat[0][0]+conf_mat[1][1])/len(y_test))
#VISUALIZATION
from os import system
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=690)
# Train
model.fit(x_train, y_train.ravel())
# Extract single tree
estimator = model.estimators_[5]
from sklearn.tree import export_graphviz
from sklearn import tree
# Export as dot file
dotfile = open("C:/Users/Downloads/treerf.dot", 'w')
export_graphviz(estimator, out_file="C:/Users/Downloads/treerf.dot")
dotfile.close()
system("dot -Tpng C:/Users/Downloads/treerf.dot -o C:/Users/Downloads/treerf.png")
from graphviz import Source
from IPython.display import SVG
tree.export_graphviz(estimator, out_file="C:/Users/Downloads/treerf.dot", feature_names=None)
system('C:/Users/anaconda3/Lib/site-packages/graphviz/')
tree.plot_tree(estimator,feature_names)

