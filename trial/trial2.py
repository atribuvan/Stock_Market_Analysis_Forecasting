import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
df = pd.read_csv('dataset5.csv')
# Preprocessing function
def preprocess_stock_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")  # Convert Date column
    df.set_index("Date", inplace=True)  # Set Date as index
    df = df[["Open Price", "High Price", "Low Price", "Close Price"]]  # Ensure correct column names
    df = df.sort_index()  # Ensure chronological order
    return df

# Load and preprocess data
df = preprocess_stock_data("dataset5.csv")

# Use only Close Price for LSTM
df = df[['Close Price']]

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# Function to create sequences for LSTM
def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])  # Past 'time_step' days
        y.append(data[i+time_step])    # Predict next day
    return np.array(X), np.array(y)

# Define time step (number of past days used)
time_step = 50  

# Split dataset into training and testing
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size-time_step:]  # Include last 50 days for continuity

# Create sequences
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

# Reshape for LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer for single value prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (Increase epochs if needed)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
predicted_price = model.predict(X_test)

# Inverse transform to get actual price range
predicted_price = scaler.inverse_transform(predicted_price)
actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(12,5))
plt.plot(range(len(actual_price)), actual_price, label="Actual Price", linewidth=2)
plt.plot(range(len(predicted_price)), predicted_price, label="Predicted Price", linewidth=2)
plt.legend()
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction using LSTM (Full Test Set)")
plt.show()

