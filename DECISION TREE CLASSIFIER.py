#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier as DTC
classifier = DTC(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
dtc = DTC()
dtc.fit(x_train,y_train)
#MODEL PERFORMANCE EVALUATION
from sklearn.metrics import classification_report,confusion_matrix
x_test = np.nan_to_num(x_test)
y_pred = dtc.predict(x_test)
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
dotfile = open("C:/Users/Downloads/treedf.dot", 'w')
export_graphviz(estimator, out_file="C:/Users/Downloads/treedf.dot")
dotfile.close()
system("dot -Tpng C:/Users/Downloads/treedf.dot -o C:/Users/Downloads/treedf.png")
from graphviz import Source
from IPython.display import SVG
tree.export_graphviz(estimator, out_file="C:/Users/Downloads/treedf.dot", feature_names=None)
system('C:/Users/anaconda3/Lib/site-packages/graphviz/')
tree.plot_tree(estimator,feature_names = None)
def Snippet_146_Ex_2():
    print('**Optimizing hyper-parameters of a Decision Tree model using Grid Search in Python**\n')
    # Creating an standardscaler object
    std_slc = StandardScaler()
    # Creating a pca object
    pca = decomposition.PCA()
    # Creating a DecisionTreeClassifier
    dec_tree = tree.DecisionTreeClassifier()
    # Creating a pipeline of three steps. First, standardizing the data.
    # Second, tranforming the data with PCA.
    # Third, training a Decision Tree Classifier on the data.
    pipe = Pipeline(steps=[('std_slc', std_slc), ('pca', pca), ('dec_tree', dec_tree)])
    # Creating Parameter Space
    # Creating a list of a sequence of integers from 1 to 30 (the number of features in X + 1)
    n_components = list(range(1,X.shape[1]+1,1))
    # Creating lists of parameter for Decision Tree Classifier
    criterion = ['gini', 'entropy']
    max_depth = [2,4,6,8,10,12]
    # Creating a dictionary of all the parameter options 
    # Note that we can access the parameters of steps of a pipeline by using '__’
    parameters = dict(pca__n_components=n_components,               dec_tree__criterion=criterion, dec_tree__max_depth=max_depth)
    # Conducting Parameter Optmization With Pipeline
    # Creating a grid search object
    clf_GS = GridSearchCV(pipe, parameters)
    # Fitting the grid search
    clf_GS.fit(X, Y)
    # Viewing The Best Parameters
    print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
    print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
    print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
    print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])

