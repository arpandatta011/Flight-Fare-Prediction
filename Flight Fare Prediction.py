#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


Flight=pd.read_csv('Data_Train.csv')


# In[3]:


Flight.head()


# In[4]:


Flight['Journey_Day']=Flight['Date_of_Journey'].apply(lambda x: x.split('/')[0])
Flight['Journey_month']=Flight['Date_of_Journey'].apply(lambda x: x.split('/')[1])


# In[6]:


Flight.head()


# In[7]:


Flight.drop('Date_of_Journey',axis=1,inplace=True)


# In[9]:


Flight.isnull().mean()


# In[11]:


Flight.dropna(inplace=True)


# In[13]:


Flight.columns


# In[14]:


Flight['Dep_Hour']=Flight['Dep_Time'].apply(lambda x: x.split(':')[0])
Flight['Dep_Min']=Flight['Dep_Time'].apply(lambda x: x.split(':')[1])


# In[15]:


Flight.head()


# In[16]:


Flight.drop('Dep_Time',axis=1,inplace=True)
Flight.head()


# In[20]:


Flight['Arrival_Hour']=pd.to_datetime(Flight['Arrival_Time']).dt.hour  
Flight['Arrival_min']=pd.to_datetime(Flight['Arrival_Time']).dt.minute


# In[21]:


Flight.drop('Arrival_Time',axis=1,inplace=True)


# In[23]:


duration=list(Flight['Duration'])


# In[25]:


for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if 'h' in duration[i]:
            duration[i]=duration[i].strip() + " 0m "
        else:
            duration[i]=' 0h ' + duration[i]      
duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1]))
Flight['Duration_hours']=duration_hours
Flight['Duration_mins']=duration_mins


# In[26]:


Flight['Duration_hours']=duration_hours
Flight['Duration_mins']=duration_mins


# In[27]:


Flight.head()


# In[28]:


Flight.drop('Duration',axis=1,inplace=True)
Flight.head()


# In[29]:


Flight['Airline'].value_counts()


# In[30]:


Airline=Flight[['Airline']]
Airline=pd.get_dummies(Airline,drop_first=True)


# In[31]:


Flight['Source'].value_counts()


# In[32]:


Source=Flight[['Source']]
Source=pd.get_dummies(Source,drop_first=True)
Source.head()


# In[33]:


Destination=Flight[['Destination']]
Destination=pd.get_dummies(Destination, drop_first=True)


# In[34]:


Flight.drop(['Route','Additional_Info'],axis=1,inplace=True)


# In[36]:


Flight['Total_Stops'].value_counts()


# In[37]:


Flight.replace({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4},inplace=True)


# In[38]:


Flight.head()


# In[40]:


Flight_final=pd.concat([Flight,Airline,Source,Destination],axis=1)
Flight_final.head()


# In[42]:


Flight_final.shape


# In[43]:


Flight_final.drop(['Airline','Source','Destination'],axis=1,inplace=True)
Flight_final.head()


# In[44]:


X=Flight_final.drop('Price',axis=1)
y=Flight_final['Price']


# In[46]:


plt.figure(figsize=(18,18))
sns.heatmap(Flight_final.corr(),annot=True,cmap='RdYlGn')
plt.show()


# In[49]:


from sklearn.ensemble import ExtraTreesRegressor
selection=ExtraTreesRegressor()
selection.fit(X,y)
print(selection.feature_importances_)


# In[50]:


plt.figure(figsize=(12,8))
feat_importances=pd.Series(selection.feature_importances_,index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[51]:


from sklearn.model_selection import train_test_split


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[56]:


from sklearn.ensemble import RandomForestRegressor
RR=RandomForestRegressor()
RR.fit(x_train,y_train)


# In[57]:


y_pred=RR.predict(x_test)


# In[58]:


RR.score(X_train,y_train) 


# In[59]:


RR.score(X_test,y_test) 


# In[ ]:




