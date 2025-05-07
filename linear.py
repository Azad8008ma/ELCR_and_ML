import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import time

# شروع زمان‌سنجی
start_time = time.time()

df= pd.read_excel('./pro_ELCR_data2.xlsx')
print(df)

x=df.drop('cr',axis='columns')
y=df['cr']
#cr=[]
#for i in y:
#    cancer_risk=((1/i)/1)/100
#    cr.append(cancer_risk)
#df.insert(11, 'cr', cr)
#df.to_excel('pro_ELCR_data2.xlsx', index=False)

print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_train_predict=np.array(model.predict(X_train))
y_test_predict=np.array(model.predict(X_test))

mae = mean_absolute_error(y_test, y_test_predict)
mse = mean_squared_error(y_test, y_test_predict)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')


# پایان زمان‌سنجی
end_time = time.time()
print("زمان اجرای برنامه:", end_time - start_time, "ثانیه")

plt.plot(X_test['rice use'],y_test,color='green' ,linestyle='none', marker="o")
plt.plot(X_test['rice use'],y_test_predict,color='red',linestyle='none',marker='o')
plt.xlabel('rice use')
plt.ylabel('cr')
plt.show()

model2=LinearRegression()
def Age_mean_slop():
    datax=np.array(df['Age']).reshape((-1,1))
    datay=np.array(df['cr']).reshape((-1,1))
    model2.fit(datax,datay)
    slop=model2.coef_[0]
    return slop

def weight_mean_slop():
    datax=np.array(df['weight']).reshape((-1,1))
    datay=np.array(df['cr']).reshape((-1,1))
    model2.fit(datax,datay)
    slop=model2.coef_[0]
    return slop

def rice_use_mean_slop():
    datax=np.array(df['rice use']).reshape((-1,1))
    datay=np.array(df['cr']).reshape((-1,1))
    model2.fit(datax,datay)
    slop=model2.coef_[0]
    return slop

def pasta_use_mean_slop():
    datax=np.array(df['pasta use']).reshape((-1,1))
    datay=np.array(df['cr']).reshape((-1,1))
    model2.fit(datax,datay)
    slop=model2.coef_[0]
    return slop

def ED_mean_slop():
    datax=np.array(df['ED']).reshape((-1,1))
    datay=np.array(df['cr']).reshape((-1,1))
    model2.fit(datax,datay)
    slop=model2.coef_[0]
    return slop

def EF_mean_slop():
    datax=np.array(df['EF']).reshape((-1,1))
    datay=np.array(df['cr']).reshape((-1,1))
    model2.fit(datax,datay)
    slop=model2.coef_[0]
    return slop

def RBA_mean_slop():
    datax=np.array(df['As_RBA']+df['Cr_RBA'] +df['Hg_RBA']+df['Cd_RBA'] +df['Pb_RBA']).reshape((-1,1))
    datay=np.array(df['cr']).reshape((-1,1))
    model2.fit(datax,datay)
    slop=model2.coef_[0]
    return slop

print(f'Age mean slop is:{Age_mean_slop()/np.mean(df["Age"])}')
print(f'weight mean slop is:{weight_mean_slop()/-np.mean(df['weight'])}')
print(f'ED mean slop is:{ED_mean_slop()/-np.mean(df['ED'])/100}')
print(f'EF mean slop is:{EF_mean_slop()/-np.mean(df["EF"])/100}')
print(f'rice use mean slop is:{rice_use_mean_slop()/1000}')
print(f'pasta use mean slop is:{pasta_use_mean_slop()/1000}')
print(f'RBA mean slop is:{RBA_mean_slop()/-np.mean(df['As_RBA']+df['Cr_RBA'] +df['Hg_RBA']+df['Cd_RBA'] +df['Pb_RBA'])}')
