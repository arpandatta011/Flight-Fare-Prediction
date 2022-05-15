
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


Flight=pd.read_csv('Data_Train.csv')


Flight.head()


Flight['Journey_Day']=Flight['Date_of_Journey'].apply(lambda x: x.split('/')[0])
Flight['Journey_month']=Flight['Date_of_Journey'].apply(lambda x: x.split('/')[1])


Flight.head()


Flight.drop('Date_of_Journey',axis=1,inplace=True)


Flight.isnull().mean()


Flight.dropna(inplace=True)


Flight.columns


Flight['Dep_Hour']=Flight['Dep_Time'].apply(lambda x: x.split(':')[0])
Flight['Dep_Min']=Flight['Dep_Time'].apply(lambda x: x.split(':')[1])


Flight.head()


Flight.drop('Dep_Time',axis=1,inplace=True)
Flight.head()



Flight['Arrival_Hour']=pd.to_datetime(Flight['Arrival_Time']).dt.hour  
Flight['Arrival_min']=pd.to_datetime(Flight['Arrival_Time']).dt.minute



Flight.drop('Arrival_Time',axis=1,inplace=True)



duration=list(Flight['Duration'])



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


Flight['Duration_hours']=duration_hours
Flight['Duration_mins']=duration_mins


Flight.head()


Flight.drop('Duration',axis=1,inplace=True)
Flight.head()


Flight['Airline'].value_counts()


Airline=Flight[['Airline']]
Airline=pd.get_dummies(Airline,drop_first=True)


Flight['Source'].value_counts()


Source=Flight[['Source']]
Source=pd.get_dummies(Source,drop_first=True)
Source.head()


Destination=Flight[['Destination']]
Destination=pd.get_dummies(Destination, drop_first=True)


Flight.drop(['Route','Additional_Info'],axis=1,inplace=True)


Flight['Total_Stops'].value_counts()


Flight.replace({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4},inplace=True)


Flight.head()


Flight_final=pd.concat([Flight,Airline,Source,Destination],axis=1)
Flight_final.head()


Flight_final.shape



Flight_final.drop(['Airline','Source','Destination'],axis=1,inplace=True)
Flight_final.head()



X=Flight_final.drop('Price',axis=1)
y=Flight_final['Price']


plt.figure(figsize=(18,18))
sns.heatmap(Flight_final.corr(),annot=True,cmap='RdYlGn')
plt.show()


from sklearn.ensemble import ExtraTreesRegressor
selection=ExtraTreesRegressor()
selection.fit(X,y)
print(selection.feature_importances_)



plt.figure(figsize=(12,8))
feat_importances=pd.Series(selection.feature_importances_,index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.ensemble import RandomForestRegressor
RR=RandomForestRegressor()
RR.fit(x_train,y_train)


y_pred=RR.predict(x_test)


RR.score(X_train,y_train) 


RR.score(X_test,y_test) 





