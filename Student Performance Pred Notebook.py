
# coding: utf-8

# In[27]:


import pandas as pd 
df=pd.read_csv('/home/sai/Downloads/student/student-mat.csv',delimiter=';')


# In[28]:


df.shape


# In[29]:


df.columns


# In[30]:


catog=['school', 'sex','address', 'famsize', 'Pstatus','Mjob', 'Fjob', 'reason','guardian','schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic']


# In[31]:


for cat in catog:
    df[cat] = pd.Categorical(df[cat])
    df[cat] = df[cat].cat.codes


# In[32]:


df.school.value_counts()


# In[33]:


df


# In[34]:


df.columns


# In[35]:


feats=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2']


# In[26]:


# from sklearn.preprocessing import StandardScaler


# scaler=StandardScaler()
# scaler.fit(df)
# df=scaler.transform(df)


# In[36]:


train=df[feats]
target=df['G3']


# In[52]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y= train_test_split(train,target,test_size=0.2)


# In[53]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,confusion_matrix
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

et=ExtraTreesRegressor(n_estimators=50,verbose=True,n_jobs=-1)
et.fit(train_x,train_y)
pred=et.predict(test_x)
print("R SQURED SCORE      :",r2_score(test_y,pred))
print("MAE                 :",mean_absolute_error(test_y,pred))
print("MSE                 :",mean_squared_error(test_y,pred))
print("RMSE                :",np.sqrt(mean_squared_error(test_y,pred)))


# In[60]:


print()


# In[61]:


test_y


# In[62]:


pred

