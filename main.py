import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("gld_price_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.strftime('%Y %m %d')
df['Date'] = df['Date'].apply(lambda x: dt.datetime.strptime(x,'%Y %m %d') if type(x)==str else pd.NaT)
df["Year"] = df["Date"].dt.year
df = df.set_index(df["Date"])
grouped = df.groupby('Year').mean(numeric_only=True)
fig2 = px.line(df,
              x="Date",
              y="GLD",
              labels = {'Date':'Year'},
              title = "Цена золота по месяцам")
fig2.show()
fig = px.box(df,
        x=df.index.month,
        y="GLD",
        color=df.index.month,
        labels = {"x" : "Месяцы"},
        title="Цена золота | Месячная статистика ")

fig.update_traces(width=0.5)
fig.show()
for i in range(len(df)):
  df['Date'][i]=df['Date'][i].timestamp()
df = df.astype('float32')
fig = px.line(df,
              x="Date",
              y="GLD",
              labels = {'Date'},
              title = "Цена золота")
fig.show()
df_corr=df.corr()
x = list(df_corr.columns)
y = list(df_corr.index)
z = np.array(df_corr)
fig1 = ff.create_annotated_heatmap(x = x,
                                  y = y,
                                  z = z,
                                  annotation_text = np.around(z, decimals=2))
fig1.show()
df817=df[df['Year'].isin(range(2008,2016))]
df18=df[df['Year'].isin(range(2017,2018))]
dfsr=df18['GLD']
x = df817[['SPX','USO','SLV','EUR/USD']]
y = df817['GLD']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=7)
#X_train = x_train.values.reshape(-1, 1, 4)
#X_test =x_test.values.reshape(-1, 1, 4)
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model=Sequential()
#model.add(Dense(256,activation='relu',input_dim=4))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train_scaled, y_train, epochs=500, verbose=0)
preds = model.predict(x_test_scaled)
print(r2_score(y_test, preds))
y_test= list(y_test)
plt.plot(y_test,color="blue",label="Действ цена")
plt.plot(preds,color="green",label="Предсказанная цена")
plt.title("Предсказанная и действительная цены")
plt.xlabel("Кол-во переменных")
plt.ylabel("Цена золота")
plt.show()
df.reset_index(drop=True)
df18=df18.drop('Date', axis=1)
df18.reset_index(drop=True)
next_year=df18[['SPX','USO','SLV','EUR/USD']]
next_year_preds = model.predict(next_year)
df18['GLD']=next_year_preds
print(df18.head)
mapedf = np.mean(np.abs((df18['GLD']) - next_year_preds / df18['GLD'])) * 100
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(df18['GLD'],dfsr )

print("Метрики работы модели:")
print("__________________________________________________________________")
print("Model Percentage Mean Absolute Error: ", mape)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R^2: ", r2)
print("Percentage Mean Absolute Error: ", mapedf)
print("__________________________________________________________________")