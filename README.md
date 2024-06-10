import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/LIUYI/Desktop/data/traffic-prediction-dataset.csv')

print(data.columns)

time_index = pd.date_range(start='2020-01-01', periods=len(data), freq='5T')
data.index = time_index

data.columns = ['Cross1', 'Cross2', 'Cross3', 'Cross4', 'Cross5', 'Cross6']

data = data.resample('5T').sum().fillna(0)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

data_scaled[data_scaled == 0] = np.nan

imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_scaled)

data_processed = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)

X = []
y = []

for i in range(len(data_processed) - 13):
    X.append(data_processed.iloc[i:i+12].values.flatten())
    y.append(data_processed.iloc[i+12].values)

X = np.array(X)
y = np.array(y)

shuffle_indices = np.arange(len(X))
np.random.shuffle(shuffle_indices)
X_shuffled = X[shuffle_indices]
y_shuffled = y[shuffle_indices]

X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.25, shuffle=False)

plt.figure(figsize=(15, 15))

def mean_absolute_error_custom(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error_custom(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r_squared_custom(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - y_mean) ** 2)
    return 1 - (ss_residual / ss_total)

for i in range(4):
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train[:, i])

    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error_custom(y_test[:, i], y_pred)
    rmse = root_mean_squared_error_custom(y_test[:, i], y_pred)
    r2 = r_squared_custom(y_test[:, i], y_pred)

    print(f'Cross {i+1} - MAE: {mae}, RMSE: {rmse}, R^2: {r2}')

    plt.subplot(4, 2, 2*i+1)
    plt.plot(range(288), y_test[:288, i], label='True Values')
    plt.plot(range(288), y_pred[:288], label='Predicted Values')
    plt.title(f'One Day Traffic Flow Prediction - Cross {i+1}')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Traffic Flow')
    plt.legend()

    plt.subplot(4, 2, 2*i+2)
    plt.plot(range(len(y_test)), y_test[:, i], label='True Values')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Values')
    plt.title(f'One Week Traffic Flow Prediction - Cross {i+1}')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Traffic Flow')
    plt.legend()

plt.tight_layout()
plt.show()
