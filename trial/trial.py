import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Preprocessing function
def preprocess_stock_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
    df.set_index("Date", inplace=True)
    df = df[["Open Price", "High Price", "Low Price", "Close Price"]]
    df = df.sort_index()
    return df

# Load and preprocess data
df = preprocess_stock_data("dataset5.csv")

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Create sequences for LSTM
def create_sequences(data, time_step=50):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step, 3])  # Predict Close Price
    return np.array(X), np.array(y)

time_step = 50
X, y = create_sequences(df_scaled, time_step)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 4)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile and train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

# Predict and rescale
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), 3)), y_pred.reshape(-1, 1)), axis=1))[:, 3]

y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), 3)), y_test.reshape(-1, 1)), axis=1))[:, 3]

# Plot results
plt.figure(figsize=(14, 5))
plt.plot(y_test_rescaled, label='Actual Price')
plt.plot(y_pred_rescaled, label='Predicted Price')
plt.legend()
plt.title('Stock Price Prediction using LSTM')
plt.show()
