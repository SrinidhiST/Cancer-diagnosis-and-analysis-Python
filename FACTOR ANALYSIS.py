#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#FACTOR ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dt = pd.read_csv(r'C:\Users\Desktop\cancer.csv')
dt.head()
np.array(list(dt.columns),dtype=object)
dt.isnull().sum()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = scaler.fit_transform(dt)
df = pd.DataFrame(data=df,columns=dt.columns)
df.head()
#BARTLETT AND KMO TEST
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity,calculate_kmo
chi2,p = calculate_bartlett_sphericity(df)
print("Chi squared value : ",chi2)
print("p value : ",p)
kmo_all,kmo_model = calculate_kmo(dt)
print("KMO Test value: ",kmo_model)
#DETERMINING NUMBER OF FACTORS AND SCREE PLOT
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(rotation = None,impute = "drop",n_factors=df.shape[1])
fa.fit(df)
ev,_ = fa.get_eigenvalues()
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid()
#EIGEN VALUES, FACTOR LOADINGS, UNIQUENESS BY FACTOR ANALYSIS
fa = FactorAnalyzer(n_factors=6,rotation='varimax')
fa.fit(dt)
with np.printoptions(suppress=True,precision=6):
    print("Factor Loadings:")
    print(pd.DataFrame(fa.loadings_,index=df.columns))
with np.printoptions(suppress=True,precision=6):
    print("Variance:")
    print(pd.DataFrame(fa.get_factor_variance(),index=['Eigen Values','Proportional Var','Cumulative Var']))
with np.printoptions(suppress=True,precision=6):
    print("Uniqueness:")
    print(pd.DataFrame(fa.get_uniquenesses(),index=df.columns,columns=['Uniqueness']))
with np.printoptions(precision=4,suppress=True):
    print("Eigen values:")
    print(pd.DataFrame(fa.get_communalities(),index=df.columns,columns=['Communalities']))
               

