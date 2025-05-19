import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# Preprocessing function
def preprocess_stock_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
    df.set_index("Date", inplace=True)
    df = df[["Open Price", "High Price", "Low Price", "Close Price"]]
    df = df.sort_index()
    return df

# Function to save the model with a unique name (timestamp)
def save_model(model):
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    model.save(os.path.join(model_dir, model_filename))
    print(f"Model saved as {model_filename}")

# Function to load the most recent model (if exists)
def load_latest_model():
    model_dir = "saved_models"
    if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
        model_files = sorted(os.listdir(model_dir), reverse=True)
        latest_model = model_files[2]  # Choose models based on index
        model = load_model(os.path.join(model_dir, latest_model))
        print(f"Loaded model: {latest_model}")
        return model
    else:
        print("No saved model found.")
        return None

# Load and preprocess data
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Preprocessed_Data/Large_Cap/ADANIPORTS.csv")) #Choose company in the cap folders
df = preprocess_stock_data(csv_path)

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

# Check if there is a previously saved model, otherwise create a new one
model = load_latest_model()

if model is None:  # No model found, create and train a new model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 4)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    save_model(model)

# Predict and rescale
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), 3)), y_pred.reshape(-1, 1)), axis=1))[:, 3]
y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), 3)), y_test.reshape(-1, 1)), axis=1))[:, 3]


# Custom accuracy
tolerance = 0.05
differences = np.abs(y_test_rescaled - y_pred_rescaled)
accuracy_mask = differences <= (tolerance * y_test_rescaled)
custom_accuracy = np.mean(accuracy_mask) * 100

print(f"Custom Accuracy (%): {custom_accuracy:.2f}%")

# Plot results
plt.figure(figsize=(14, 5))
plt.plot(y_test_rescaled, label='Actual Price')
plt.plot(y_pred_rescaled, label='Predicted Price')
plt.legend()
plt.title('Stock Price Prediction using LSTM')
plt.show()
