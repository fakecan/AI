#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Through this kernel,We are going to learn and try some of the most commonly used encoding techniques.As this competition mainly deals with encoding I hope that it would be a great time to refresh some the most common and effective encoding techniques currently in use.
# - **We will also run and test each of these encoding techniques in a simple logistic regression model and finally observe the performance of each type of encoding.**

# ![](https://media.giphy.com/media/H4DjXQXamtTiIuCcRU/giphy.gif)

# #### So,let's begin...

# - **Method 1 :** [Label encoding](#1)
# - **Method 2 :** [OnHot encoding](#2)
# - **Method 3 :** [Feature Hashing](#3)
# - **Method 4 :** [Encoding categories with dataset statistics](#4)
# - **Cyclic features :** [Encoding cyclic features](#6)
# - **Method 5:** [Target Encoding](#7)
# - **Method 6 :** [K-Fold target encoding](#8)
# - **Summary :** [Summary of model performance](#5)

# #### If you think this notebook is worth reading and has gained some knowledge from this,please consider upvoting my kernel.Your appreciation means a lot to me...

# ## Importing required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import base


# In[ ]:


df_train=pd.read_csv('./data/cat-in-the-dat/train.csv')
df_test=pd.read_csv('./data/cat-in-the-dat/test.csv')


# In[ ]:


print('train data set has got {} rows and {} columns'.format(df_train.shape[0],df_train.shape[1]))
print('test data set has got {} rows and {} columns'.format(df_test.shape[0],df_test.shape[1]))


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# ### Defining the train and target

# In[ ]:


X=df_train.drop(['target'],axis=1)
y=df_train['target']
#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


# In[ ]:


x=y.value_counts()
plt.bar(x.index,x)
plt.gca().set_xticks([0,1])
plt.title('distribution of target variable')
plt.show()


# Before getting into encoding,I will just breif you with types data variables present in this data:
# - **Binary data** : A  binary variable a variable that has only 2 values..ie 0/1
# - **Categorical data** : A categorical variable is a variable that can take some limited number of values.for example,day of the week.It can be one of 1,2,3,4,5,6,7 only.
# - **Ordinal data** : An ordinal variable is a categorical variable that has some order associated with it.for example,the ratings that are given to a movie by a user.
# - **Nominal data** :  Nominal value is a variable that has no numerical importance,such as occupation,person name etc..
# - **Timeseries data** : Time series data has a temporal value attached to it, so this would be something like a date or a time stamp that you can look for trends in time.
# 
# 

# ## Method 1: Label encoding <a id='1'></a>
# In this method we change every categorical data to a number.That is each type will be subtuted by a number.for example we will substitute 1 for Grandmaster,2 for master ,3 for expert etc..
# For implementing this we will first import *Labelencoder* from  *sklearn* module.

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# Now we will do these three steps to label encode our data:
# - Initialize the labelencoder class
# - Call the fit() method to fit the data
# - Transform data to labelencoded data

# In[ ]:


train=pd.DataFrame()
label=LabelEncoder()
for c in  X.columns:
    if(X[c].dtype=='object'):
        train[c]=label.fit_transform(X[c])
    else:
        train[c]=X[c]
        
train.head(3)    


# Here you can see the label encoded output train data.We will check the shape of train data now and verify that there is no change in the number of columns.

# In[ ]:



print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))


# ### Logistic regression

# In[ ]:


def logistic(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    y_pre=lr.predict(X_test)
    print('Accuracy : ',accuracy_score(y_test,y_pre))


# In[ ]:


logistic(train,y)


# ## Method 2 : On hot encoding  <a id='2'></a>
# Our second method is encoding each category as a one hot encoding (OHE) vector (or dummy variables). OHE is a representation method that takes each category value and turns it into a binary vector of size |i|(number of values in category i) where all columns are equal to zero besides the category column. Here is a little example:   
# 
# 
# ![](https://miro.medium.com/max/878/1*WXpoiS7HXRC-uwJPYsy1Dg.png)
# 
# To implement on-hot encoding we will use *get_dummies()* function in *pandas*.
# 
# 

# In[ ]:


#train=pd.get_dummies(X).astype(np.int8)
#print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))


# This produces output as a pandas dataframe.Alternatively we can use *OneHotEncoder()* method available in* sklearn* to convert out data to on-hot encoded data.But this method produces a sparse metrix.The advantage of this methos is that is uses very less memory/cpu resourses.
# To do that,we need to :
# - Import OneHotEncoder from sklean.preprocessing
# - Initialize the OneHotEncoder
# - Fit and then transform our data

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()

one.fit(X)
train=one.transform(X)

print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))


# In[ ]:


logistic(train,y)


# ## Method 3 : Feature hashing (a.k.a the hashing trick)  <a id='3'></a>

# Feature hashing is a very cool technique to represent categories in a “one hot encoding style” as a sparse matrix but with a much lower dimensions. In feature hashing we apply a hashing function to the category and then represent it by its indices. for example, if we choose a dimension of 5 to represent “New York” we will calculate H(New York) mod 5 = 3 (for example) so New York representation will be (0,0,1,0,0).

'''
# In[ ]:


from sklearn.feature_extraction import FeatureHasher
X_train_hash=X.copy()
for c in X.columns:
    X_train_hash[c]=X[c].astype('str')      
hashing=FeatureHasher(input_type='string')
train=hashing.transform(X_train_hash.values)


# In[ ]:



print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))


# In[ ]:


logistic(train,y)


# ## Method 4 :Encoding categories with dataset statistics  <a id='4'></a>

#  Now we will try to give our models a numeric representation for every category with a small number of columns but with an encoding that will put similar categories close to each other. The easiest way to do it is replace every category with the number of times that we saw it in the dataset. This way if New York and New Jersey are both big cities, they will probably both appear many times in our dataset and the model will know that they are similar.

# In[ ]:


X_train_stat=X.copy()
for c in X_train_stat.columns:
    if(X_train_stat[c].dtype=='object'):
        X_train_stat[c]=X_train_stat[c].astype('category')
        counts=X_train_stat[c].value_counts()
        counts=counts.sort_index()
        counts=counts.fillna(0)
        counts += np.random.rand(len(counts))/1000
        X_train_stat[c].cat.categories=counts
    
        
        


# In[ ]:


X_train_stat.head(3)


# In[ ]:


print('train data set has got {} rows and {} columns'.format(X_train_stat.shape[0],X_train_stat.shape[1]))
        


# In[ ]:


logistic(X_train_stat,y)


# ## Encoding cyclic features  <a id='6'></a>
# ![](https://miro.medium.com/max/343/1*70cevmU8wNggGJEdLam1lw.png)
# 
# Some of our features are cyclic in nature.ie day,month etc.
# 
# A common method for encoding cyclical data is to transform the data into two dimensions using a sine and consine transformation.
# 
# 

# In[ ]:


X_train_cyclic=X.copy()
columns=['day','month']
for col in columns:
    X_train_cyclic[col+'_sin']=np.sin((2*np.pi*X_train_cyclic[col])/max(X_train_cyclic[col]))
    X_train_cyclic[col+'_cos']=np.cos((2*np.pi*X_train_cyclic[col])/max(X_train_cyclic[col]))
X_train_cyclic=X_train_cyclic.drop(columns,axis=1)

X_train_cyclic[['day_sin','day_cos']].head(3)


# Now we will use OnHotEncoder to encode other variables,then feed the data to our model.

# In[ ]:


one=OneHotEncoder()

one.fit(X_train_cyclic)
train=one.transform(X_train_cyclic)

print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))


# In[ ]:


logistic(train,y)


# ## Method 5 : Target encoding <a id='7'></a>
#  		
# Target-based encoding is numerization of categorical variables via target. In this method, we replace the categorical variable with just one new numerical variable and replace each category of the categorical variable with its corresponding probability of the target (if categorical) or average of the target (if numerical). The main drawbacks of this method are its dependency to the distribution of the target, and its lower predictability power compare to the binary encoding method.
# 
# for example,
# <table style="width : 20%">
#     <tr>
#     <th>Country</th>
#     <th>Target</th>
#     </tr>
#     <tr>
#     <td>India</td>
#     <td>1</td>
#     </tr>
#     <tr>
#     <td>China</td>
#     <td>0</td>
#     </tr>
#     <tr>
#     <td>India</td>
#     <td>0</td>
#     </tr>
#     <tr>
#     <td>China</td>
#     <td>1</td>
#     </tr>
#     </tr>
#     <tr>
#     <td>India</td>
#     <td>1</td>
#     </tr>
# </table>
# 
# 

# Encoding for India = [Number of true targets under the label India/ Total Number of targets under the label India] 
# which is 2/3 = 0.66
# 
# <table style="width : 20%">
#     <tr>
#     <th>Country</th>
#     <th>Target</th>
#     </tr>
#     <tr>
#     <td>India</td>
#     <td>0.66</td>
#     </tr>
#     <tr>
#     <td>China</td>
#     <td>0.5</td>
#     </tr>
# </table>
# 
# 

# In[ ]:


X_target=df_train.copy()
X_target['day']=X_target['day'].astype('object')
X_target['month']=X_target['month'].astype('object')
for col in X_target.columns:
    if (X_target[col].dtype=='object'):
        target= dict ( X_target.groupby(col)['target'].agg('sum')/X_target.groupby(col)['target'].agg('count'))
        X_target[col]=X_target[col].replace(target).values
        
    
    

X_target.head(4)


# In[ ]:


logistic(X_target.drop('target',axis=1),y)


# ### K-Fold target encoding <a id='8' ></a>
# 
# k-fold target encoding can be applied to reduce the overfitting. In this method, we divide the dataset into the k-folds, here we consider 5 folds. Fig.3 shows the first round of the 5 fold cross-validation. We calculate mean-target for fold 2, 3, 4 and 5 and we use the calculated values, mean_A = 0.556 and mean_B = 0.285 to estimate mean encoding for the fold-1.

# ![](https://miro.medium.com/max/1955/1*ZKD4eZXzd_FdN0SQDszFVQ.png)

# In[ ]:


X['target']=y
cols=X.drop(['target','id'],axis=1).columns


# In[ ]:


X_fold=X.copy()
X_fold[['ord_0','day','month']]=X_fold[['ord_0','day','month']].astype('object')
X_fold[['bin_3','bin_4']]=X_fold[['bin_3','bin_4']].replace({'Y':1,'N':0,'T':1,"F":0})
kf = KFold(n_splits = 5, shuffle = False, random_state=2019)
for train_ind,val_ind in kf.split(X):
    for col in cols:
        if(X_fold[col].dtype=='object'):
            replaced=dict(X.iloc[train_ind][[col,'target']].groupby(col)['target'].mean())
            X_fold.loc[val_ind,col]=X_fold.iloc[val_ind][col].replace(replaced).values

            


# In[ ]:


X_fold.head()


# # Summary <a id='5'></a>
# 
# Here you can see the summary of our model performance against each of the encoding techniques we have used.
# It is clear that OnHotEncoder together with cyclic feature encoding yielded maximum accuracy.
# 
# <table style="width : 50%">
#     <tr>
#     <th>Encoding</th>
#     <th>Score</th>
#     </tr>
#     <tr>
#     <td>Label Encoding</td>
#     <td>0.692</td>
#     </tr>
#     <tr>
#     <td>OnHotEncoder</td>
#     <td>0.759</td>
#     </tr>
#     <tr>
#     <td>Feature Hashing</td>
#     <td>0.751</td>
#     </tr>
#     <tr>
#     <td>Dataset statistic encoding</td>
#     <td>0.694</td>
#     </tr>
#     </tr>
#     <tr>
#     <td>Cyclic + OnHotEncoding</td>
#     <td>0.759</td>
#     </tr>
#     </tr>
#     <tr>
#     <td>Target encoding</td>
#     <td>0.694</td>
#     </tr>
#     
# </table>
#     

# **You can try two or more of this approaches together,and encode the dataset in suitable way to acheive higher accuracy**
# 

# In[ ]:





# ### Please do an upvote ^ if you like my work..
# #### please don't hesitate to provide your suggestions in the comment sections.Thank you
# 

# ### Kernel under construction !!!
'''