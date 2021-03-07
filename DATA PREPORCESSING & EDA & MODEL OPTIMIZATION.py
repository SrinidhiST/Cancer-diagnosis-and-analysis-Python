#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings('ignore')
dt = pd.read_csv(r'cancer_dt.csv',index_col = 0)
dt.head()
print("Sum of null values in dataset")
dt.isnull().sum()
dt['level']= dt["level"].replace("Low", "0")
dt['level']= dt["level"].replace("Medium", "1")
dt['level']= dt["level"].replace("High", "2")
x=dt.iloc[:,:-1].values
y=dt.iloc[:,-1].values


# In[ ]:


import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
level_level = dt.groupby('level').agg('count')
print(level_level)
type_labels = level_level.age.sort_values().index 
type_counts = level_level.age.sort_values()
plt.figure(1, figsize=(20,10)) 
the_grid = GridSpec(2, 2)
cmap = plt.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0, 1, 8)]
lg = plt.subplot(the_grid[0, 1], aspect=1, title='Levels of Cancer patients')
label_ids = plt.pie(type_counts, labels=type_labels, startangle = 90,autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()


# In[ ]:


#Using Pearson Correlation
plt.figure(figsize=(12,12))
cor = dt.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#FEATURE SELECTION USING EMBEDDED METHOD
x=dt.drop("level",1)
y=dt["level"]
from sklearn.linear_model import LassoCV
reg = LassoCV()
reg.fit(x, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x,y))
coef = pd.Series(reg.coef_, index = x.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp = imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
#Correlation with output variable
#Selecting highly correlated features
corr_matrix = dt.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]
# Drop features 
dt.drop(dt[to_drop], axis=1)
#Finding outliers between least correlated variable and level variable
#Age vs Level
ax = dt['age'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Age', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Gender vs Level
ax = dt['gender'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Gender', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Air pollution vs Level
ax = dt['air_pollution'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Air Pollution', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Weight Loss vs Level
ax = dt['weight_loss'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Weight loss', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Wheezing vs Level
ax = dt['wheezing'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Wheezing', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Swallowing difficulty vs Level
ax = dt['swallowing_difficulty'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Swallowing difficulty', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Clubbing of finger nails vs Level
ax = dt['clubbing_nails'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Clubbing of finger nails', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Frequent cold vs Level
ax = dt['frequent_cold'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Frequent cold', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Snoring vs Level
ax = dt['snoring'].groupby(dt['level']).value_counts().plot(kind='bar')
ax.set_title('Frequency Distribution of Level with respect to Snoring', fontsize=20)
for item in ax.get_xticklabels():
    item.set_rotation(45)
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
#Calculating chi2 and p value
from sklearn.preprocessing import MinMaxScaler
minmaxer = MinMaxScaler(feature_range=(1,10))
minmaxed_x = minmaxer.fit_transform(x)
from sklearn.feature_selection import chi2
chi_value,pval = chi2(minmaxed_x,y)
pval = np.round(pval,decimals=3)
with np.printoptions(precision=4,suppress=True):
  print(pd.DataFrame(np.concatenate((chi_value.reshape(-1,1),pval.reshape(-1,1)),axis=1),index = dt.columns[:-1],columns=['chi2 val','pval']))
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
level_encoded=le.fit_transform(dt.level)
lae = level_encoded
print(lae)
dt.drop('level', axis=1)
lae = pd.DataFrame(lae)
dt = pd.concat([dt,lae],axis=1)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
def print_scores(model, X_train, Y_train, cv, scoring):
    print('Cross validation scores:', cross_val_score(model, X_train, Y_train, cv=cv, scoring=scoring, n_jobs=-1) )
    print( 'Mean of scores:', np.mean( cross_val_score(model, X_train, Y_train, cv=cv, scoring=scoring, n_jobs=-1) ) )
    print( 'Variance:', np.std( cross_val_score(model, X_train, Y_train, cv=cv, scoring=scoring, n_jobs=-1) ) )
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1, 9))
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42, stratify=y)

