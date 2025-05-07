import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import subprocess

result = subprocess.run(['python', 'filters.py'], capture_output=True, text=True)
print(result.stdout)

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

model=MLPRegressor()
model.fit(X_train,y_train)

y_train_predict=np.array(model.predict(X_train))
y_test_predict=np.array(model.predict(X_test))

mae = mean_absolute_error(y_test, y_test_predict)
mse = mean_squared_error(y_test, y_test_predict)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')



plt.plot(X_test['EF'],y_test,color='green' ,linestyle='none', marker="o")
plt.plot(X_test['EF'],y_test_predict,color='red',linestyle='none',marker='o')
plt.xlabel('Age')
plt.ylabel('cr')
plt.show()

